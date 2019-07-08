from glob import glob
from os.path import basename
from concurrent import futures
from os import system
import time
import logging
import sys
import signal
from subprocess import call

import grpc
import click
import numpy as np
import fasttext
import gevent
from gevent.pool import Pool
from soyspacing.countbase import CountSpace

import fasttextserver_pb2 as pb2
import fasttextserver_pb2_grpc as pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class FasttextServer(pb2_grpc.FasttextServicer):

    TYPE_WORD = 'word'
    TYPE_SENTENCE = 'sentence'
    TYPE_PREDICT = 'predict'

    def __init__(self, model_path, spacing_model_path=None, default_version='default'):
        self._model_path = model_path
        self._default_version = {self.TYPE_WORD: default_version,
                self.TYPE_SENTENCE: default_version, self.TYPE_PREDICT: default_version}
        self._word_model = {}
        self._predict_model = {}
        self._sentence_model = {}
        self._pool = Pool(16)
        threads = []

        if spacing_model_path:
            def load_spacing_model(spacing_model_path):
                logging.info('soyspacing model loading... from %s', spacing_model_path)
                start_time = time.time()
                self._spacing_model = CountSpace()
                self._spacing_model.load_model(spacing_model_path, json_format=False)
                logging.info('soyspacing model loaded, %.2f s', (time.time() - start_time))
            threads.append(gevent.spawn(load_spacing_model, spacing_model_path))
        else:
            self._spacing_model = None

        def load_model_fn(model_type, filepath):
            logging.info('%s model loading... from %s', model_type, filepath)
            start_time = time.time()
            version = basename(filepath)[:-4]
            load_model = getattr(self, '_load_%s_model' % model_type)
            load_model(filepath, version)
            logging.info('%s model loaded, version: %s, %.2f s',
                    model_type, version, (time.time() - start_time))

        for model_type in [self.TYPE_WORD, self.TYPE_SENTENCE, self.TYPE_PREDICT]:
            for filepath in glob('%s/%s/*.bin' % (self._model_path, model_type)):
                threads.append(gevent.spawn(load_model_fn, model_type, filepath))

        gevent.joinall(threads)

    def WordEmbedding(self, request, context): 
        logging.debug('word_embedding request: %s, %s', request.sentence, request.version)
        return self._word_embedding(request.sentence, request.version, request.spacing)

    def _word_embedding(self, sentence, version, spacing):
        if spacing and self._spacing_model:
            sentence, _ = self._spacing_model.correct(sentence)
        embeddings, words = self._get_embeddings(sentence, version)
        return pb2.WordEmbeddingResponse(embeddings=embeddings, words=words)

    def MultiWordEmbeddings(self, request, context): 
        logging.debug('multi_word_embedding request: %s sentences, %s, %s', len(request.sentences), request.version, request.spacing)

        def word_embedding(sentence):
            return self._word_embedding(sentence, request.version, request.spacing)

        items = self._pool.map(word_embedding, request.sentences)
        return pb2.MultiWordEmbeddingsResponse(items=items)

    def SentenceEmbedding(self, request, context):
        logging.debug('sentence_embedding request: %s, %s', request.sentence, request.version)
        embeddings = self._get_sentence_embeddings(request.sentence, request.version)
        return pb2.SentenceEmbeddingResponse(embeddings=embeddings)

    def Predict(self, request, context):
        logging.debug('predict request: %s, %s', request.sentence, request.version)
        labels, probs = self._predict(request.sentence, request.version, request.limit)
        return pb2.PredictResponse(labels=labels, probs=probs)

    def Reload(self, request, context): 
        logging.debug('reload request: %s, %s, %s', \
                request.model_type, request.version, request.filepath)
        model_type = request.model_type
        version = request.version or self._default_version[model_type]
        model_filepath = '%s/%s/%s.bin' % (self._model_path, model_type, version)
        if request.filepath and request.filepath.startswith('s3://'):
            call('aws s3 cp %s %s' % (request.filepath, model_filepath), shell=True)

        load_model = getattr(self, '_load_%s_model' % request.model_type)
        load_model(model_filepath, request.version)
        return pb2.Response(message='Reloaded: %s, %s' % \
                (request.model_type, request.version))

    def stop(self):
        pass

    def _load_word_model(self, model_filepath, version):
        model = fasttext.load_model(model_filepath)
        self._word_model[version] = model
        self._get_embeddings('test', version) # pre loading

    def _load_sentence_model(self, model_filepath, version=None):
        model = fasttext.load_model(model_filepath)
        self._sentence_model[version] = model
        self._get_sentence_embeddings('test', version) # pre loading

    def _load_predict_model(self, model_filepath, version):
        model = fasttext.load_model(model_filepath)
        self._predict_model[version] = model
        self._predict('test', version) # pre loading

    def _predict(self, sentence, version=None, k=10):
        version = version or self._default_version[self.TYPE_PREDICT]
        sentence = sentence.strip()
        k = k or 10
        if sentence.find('\n') > -1:
            raise ValueError('sentence must not contain new line(\\n)')
        if k < 1:
            raise ValueError('k(%d) must be greater than zero' % k)

        labels, probs = self._predict_model[version].predict(sentence, k=k)
        labels = [label[9:] for label in labels]
        return labels, probs

    def _get_sentence_embeddings(self, sentence, version=None):
        sentence = sentence.strip()
        if sentence.find('\n') > -1:
            raise ValueError('sentence must not contain new line(\\n)')

        version = version or self._default_version[self.TYPE_SENTENCE]
        return self._sentence_model[version].get_sentence_vector(sentence)

    def _get_embeddings(self, sentence, version=None):
        sentence = sentence.strip()
        version = version or self._default_version[self.TYPE_WORD]
        if sentence.find('\n') > -1:
            raise ValueError('sentence must not contain new line(\\n)')
        words = sentence.split()
        embeddings = []

        for word in words:
            embedding = self._word_model[version].get_word_vector(word)
            embeddings.append(embedding)
        if embeddings:
            embeddings = np.concatenate(tuple(embeddings)).tolist()

        return embeddings, words


@click.command()
@click.option('--model-path', default='models', help='model path')
@click.option('--spacing-model-path', default='models/soyspacing', help='soyspacing model trained filepath')
@click.option('--s3-model-path', help='log filepath')
@click.option('--log', help='log filepath')
@click.option('--debug', is_flag=True, help='debug')
def serve(model_path, spacing_model_path, s3_model_path, log, debug):
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

    if s3_model_path:
        logging.info('s3 model fetching...')
        start_time = time.time()
        result = call('aws s3 cp %s %s --recursive' % (s3_model_path, model_path), shell=True)
        if result == 255:
            error = 's3 model fetch error, path: %s' % s3_model_path
            logging.error(error)
            raise Exception(error)
        logging.info('s3 model fetched, %.2f s' % (time.time() - start_time))

    logging.info('server loading...')
    start_time = time.time()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    fasttext_server = FasttextServer(model_path=model_path, spacing_model_path=spacing_model_path)
    pb2_grpc.add_FasttextServicer_to_server(fasttext_server, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logging.info('server started, loading time %.2f s' % (time.time() - start_time))

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
