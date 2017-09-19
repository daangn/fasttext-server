from concurrent import futures
import time
import logging
import sys
import signal
from subprocess import Popen, PIPE

import grpc
import click

import fasttextserver_pb2 as pb2
import fasttextserver_pb2_grpc as pb2_grpc

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class FasttextServer(pb2_grpc.FasttextServicer):

    def __init__(self, word_model_filepath=None, sentence_model_filepath=None, predict_model_filepath=None):
        self.proc = None
        self.proc_sentence = None
        self.proc_predict = None
        self._word_model_filepath = word_model_filepath
        self._sentence_model_filepath = sentence_model_filepath
        self._predict_model_filepath = predict_model_filepath
        if predict_model_filepath:
            self._load_predict_model(self._predict_model_filepath)
            logging.debug('predict model loaded')
        if sentence_model_filepath:
            self._load_sentence_model(self._sentence_model_filepath)
            logging.debug('sentence model loaded')
        if word_model_filepath:
            self._load_word_model(word_model_filepath)
            logging.debug('word model loaded')

    def WordEmbedding(self, request, context): 
        logging.debug('word_embedding request: %s', request.sentence)
        embeddings, words = self._get_embeddings(request.sentence)
        return pb2.WordEmbeddingResponse(embeddings=embeddings, words=words)

    def SentenceEmbedding(self, request, context):
        logging.debug('sentence_embedding request: %s', request.sentence)
        embeddings = self._get_sentence_embeddings(request.sentence)
        return pb2.SentenceEmbeddingResponse(embeddings=embeddings)

    def Predict(self, request, context):
        logging.debug('predict request: %s', request.sentence)
        labels, probs = self._predict(request.sentence)
        return pb2.PredictResponse(labels=labels, probs=probs)

    def Reload(self, request, context): 
        key = '_%s_model_filepath' % request.model_type
        if request.filepath:
            setattr(self, key, request.filepath)
        model_filepath = getattr(self, key, None)
        load_model = getattr(self, '_load_%s_model' % request.model_type)
        load_model(model_filepath)
        return pb2.Response(message='Reloaded: %s, %s' % (request.model_type, request.filepath))

    def _load_word_model(self, model_filepath):
        pre_proc = self.proc
        self.proc = Popen(["fasttext", 'print-word-vectors', model_filepath],
                stdout=PIPE, stdin=PIPE, bufsize=1, universal_newlines=True)
        self._get_embeddings('test') # pre loading
        if pre_proc:
            pre_proc.kill()

    def _load_sentence_model(self, model_filepath):
        logging.debug('sentence model filepath: %s', model_filepath)
        pre_proc = self.proc_sentence
        self.proc_sentence = Popen(["fasttext", 'print-sentence-vectors', model_filepath],
                stdout=PIPE, stdin=PIPE, bufsize=1, universal_newlines=True)
        self._get_sentence_embeddings('test') # pre loading
        if pre_proc:
            pre_proc.kill()

    def _load_predict_model(self, model_filepath):
        logging.debug('predict model filepath: %s', model_filepath)
        pre_proc = self.proc_predict
        self.proc_predict = Popen(["fasttext", 'predict-prob', model_filepath, '-', '1000'],
                stdout=PIPE, stdin=PIPE, bufsize=1, universal_newlines=True)
        self._predict('test') # pre loading
        if pre_proc:
            pre_proc.kill()

    def _predict(self, sentence):
        sentence = sentence.strip()
        if sentence.find('\n') > -1:
            raise ValueError('sentence must not contain new line(\\n)')

        proc = self.proc_predict
        proc.stdin.write("%s\n" % sentence)
        line = proc.stdout.readline()
        tokens = line.rstrip().split()
        labels = [x[9:] for x in tokens[::2]]
        probs = [float(x) for x in tokens[1::2]]
        return labels, probs

    def _get_sentence_embeddings(self, sentence):
        sentence = sentence.strip()
        if sentence.find('\n') > -1:
            raise ValueError('sentence must not contain new line(\\n)')

        proc = self.proc_sentence
        proc.stdin.write("%s\n" % sentence)
        line = proc.stdout.readline()
        tokens = line.rstrip().split()
        embedding = [float(x) for x in tokens]
        return embedding

    def _get_embeddings(self, sentence):
        sentence = sentence.strip()
        if sentence.find('\n') > -1:
            raise ValueError('sentence must not contain new line(\\n)')
        words = sentence.split()
        words_count = len(words)

        self.proc.stdin.write("%s\n" % sentence)
        embeddings = []
        words = []
        for i in range(words_count):
            line = self.proc.stdout.readline()
            tokens = line.rstrip().split()
            if len(tokens) < 1:
                continue
            words.append(tokens[0])
            embedding = [float(x) for x in tokens[1:]]
            embeddings += embedding
        return embeddings, words

    def stop(self):
        if self.proc:
            self.proc.kill()
        if self.proc_sentence:
            self.proc_sentence.kill()
        if self.proc_predict:
            self.proc_predict.kill()


@click.command()
@click.option('--word_model', default=None, help='word model filepath')
@click.option('--sentence_model', default='models/sentence.bin', help='sentence model filepath')
@click.option('--predict_model', default='models/sentence.bin', help='sentence model filepath')
@click.option('--log', help='log filepath')
@click.option('--debug', is_flag=True, help='debug')
def serve(word_model, sentence_model, predict_model, log, debug):
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
    fasttext_server = FasttextServer(word_model_filepath=word_model,
            sentence_model_filepath=sentence_model,
            predict_model_filepath=predict_model)
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
