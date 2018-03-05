from glob import glob
from os.path import basename
from concurrent import futures
import time
import logging
import sys
import signal
import threading
from subprocess import Popen, PIPE

import grpc
import click
import numpy as np
import fastText
from soyspacing.countbase import CountSpace

import fasttextserver_pb2 as pb2
import fasttextserver_pb2_grpc as pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class FasttextServer(pb2_grpc.FasttextServicer):

    TYPE_WORD = 'word'
    TYPE_SENTENCE = 'sentence'
    TYPE_PREDICT = 'predict'
    LOCK = threading.Lock()

    def __init__(self, model_path, spacing_model_path=None, default_version='default'):
        self._model_path = model_path
        self._default_version = {self.TYPE_WORD: default_version,
                self.TYPE_SENTENCE: default_version, self.TYPE_PREDICT: default_version}
        self._proc = {self.TYPE_WORD: {}, self.TYPE_SENTENCE: {}, self.TYPE_PREDICT: {}}

        for model_type in [self.TYPE_WORD, self.TYPE_SENTENCE, self.TYPE_PREDICT]:
            for filepath in glob('%s/%s/*.bin' % (self._model_path, model_type)):
                version = basename(filepath)[:-4]
                load_model = getattr(self, '_load_%s_model' % model_type)
                load_model(filepath, version)
                logging.debug('%s model loaded, version: %s', model_type, version)

        if spacing_model_path:
            logging.debug('soyspacing model loading... from %s', spacing_model_path)
            self._spacing_model = CountSpace()
            self._spacing_model.load_model(spacing_model_path, json_format=False)
            logging.debug('soyspacing model loaded')
        else:
            self._spacing_model = None

    def WordEmbedding(self, request, context): 
        logging.debug('word_embedding request: %s, %s', request.sentence, request.version)
        with self.LOCK:
            sentence = request.sentence
            if self._spacing_model:
                sentence, _ = self._spacing_model.correct(request.sentence)
            embeddings, words = self._get_embeddings(sentence, request.version)
        return pb2.WordEmbeddingResponse(embeddings=embeddings, words=words)

    def SentenceEmbedding(self, request, context):
        logging.debug('sentence_embedding request: %s, %s', request.sentence, request.version)
        embeddings = self._get_sentence_embeddings(request.sentence, request.version)
        return pb2.SentenceEmbeddingResponse(embeddings=embeddings)

    def Predict(self, request, context):
        logging.debug('predict request: %s, %s', request.sentence, request.version)
        labels, probs = self._predict(request.sentence, request.version)
        return pb2.PredictResponse(labels=labels, probs=probs)

    def Reload(self, request, context): 
        logging.debug('reload request: %s, %s', \
                request.model_type, request.version)
        model_type = request.model_type
        version = request.version or self._default_version[model_type]
        model_filepath = '%s/%s/%s.bin' % (self._model_path, model_type, version)
        load_model = getattr(self, '_load_%s_model' % request.model_type)
        load_model(model_filepath, request.version)
        return pb2.Response(message='Reloaded: %s, %s' % \
                (request.model_type, request.version))

    def stop(self):
        for model_type in self._proc:
            map(lambda x: x.kill(), self._proc[model_type].values())

    def _get_process(self, model_type, version=None):
        version = version or self._default_version[model_type]
        if version not in self._proc[model_type]:
            return None
        return self._proc[model_type][version]

    def _set_process(self, proc, model_type, version=None):
        version = version or self._default_version[model_type]
        pre_proc = self._get_process(model_type, version)
        self._proc[model_type][version] = proc
        if pre_proc:
            pre_proc.kill()

    def _load_word_model(self, model_filepath, version=None):
        self._word_model = fastText.load_model(model_filepath)
        self._get_embeddings('test', version) # pre loading

    def _load_sentence_model(self, model_filepath, version=None):
        logging.debug('sentence model filepath: %s', model_filepath)
        proc = Popen(["fasttext", 'print-sentence-vectors', model_filepath],
                stdout=PIPE, stdin=PIPE, bufsize=1, universal_newlines=True)
        self._set_process(proc, self.TYPE_SENTENCE, version)
        self._get_sentence_embeddings('test', version) # pre loading

    def _load_predict_model(self, model_filepath, version=None):
        logging.debug('predict model filepath: %s', model_filepath)
        proc = Popen(["fasttext", 'predict-prob', model_filepath, '-', '1000'],
                stdout=PIPE, stdin=PIPE, bufsize=1, universal_newlines=True)
        self._set_process(proc, self.TYPE_PREDICT, version)
        self._predict('test', version) # pre loading

    def _predict(self, sentence, version=None):
        sentence = sentence.strip()
        if sentence.find('\n') > -1:
            raise ValueError('sentence must not contain new line(\\n)')

        proc = self._get_process(self.TYPE_PREDICT, version)
        if not proc:
            raise Exception('no process for version %s, type %s' % (version, self.TYPE_PREDICT))
        proc.stdin.write("%s\n" % sentence)
        line = proc.stdout.readline()
        tokens = line.rstrip().split()
        labels = [x[9:] for x in tokens[::2]]
        probs = [float(x) for x in tokens[1::2]]
        return labels, probs

    def _get_sentence_embeddings(self, sentence, version=None):
        sentence = sentence.strip()
        if sentence.find('\n') > -1:
            raise ValueError('sentence must not contain new line(\\n)')

        proc = self._get_process(self.TYPE_SENTENCE, version)
        if not proc:
            raise Exception('no process for version %s, type %s' % (version, self.TYPE_SENTENCE))
        proc.stdin.write("%s\n" % sentence)
        line = proc.stdout.readline()
        tokens = line.rstrip().split()
        embedding = [float(x) for x in tokens]
        return embedding

    def _get_embeddings(self, sentence, version=None):
        sentence = sentence.strip()
        if sentence.find('\n') > -1:
            raise ValueError('sentence must not contain new line(\\n)')
        words = sentence.split()
        embeddings = []

        for word in words:
            embedding = self._word_model.get_word_vector(word)
            embeddings.append(embedding)
        if embeddings:
            embeddings = np.concatenate(tuple(embeddings)).tolist()

        return embeddings, words


@click.command()
@click.option('--model_path', default='models', help='model path')
@click.option('--spacing_model_path', help='soyspacing model trained filepath')
@click.option('--log', help='log filepath')
@click.option('--debug', is_flag=True, help='debug')
def serve(model_path, log, spacing_model_path, debug):
    if log:
        handler = logging.FileHandler(filename=log)
    else:
        handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s - %(message)s')
    handler.setFormatter(formatter)
    root = logging.getLogger()
    level = debug and logging.DEBUG or logging.INFO
    root.setLevel(level)
    root.addHandler(handler)

    logging.info('server loading...')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    fasttext_server = FasttextServer(model_path=model_path, spacing_model_path=spacing_model_path)
    pb2_grpc.add_FasttextServicer_to_server(fasttext_server, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logging.info('server started')

    # for docker heath check
    with open('/tmp/status', 'w') as f:
        f.write('started')

    def stop_serve(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, stop_serve)
    signal.signal(signal.SIGTERM, stop_serve)

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
        fasttext_server.stop()
        logging.info('server stopped')

if __name__ == '__main__':
    serve()
