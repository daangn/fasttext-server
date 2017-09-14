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

    def __init__(self):
        self.proc = Popen(["fasttext", 'print-word-vectors', 'models/skipgram.bin'],
                stdout=PIPE, stdin=PIPE, bufsize=1, universal_newlines=True)
        self.get_embeddings('test') # pre loading

    def WordEmbedding(self, request, context): 
        embeddings, words = self.get_embeddings(request.sentence)
        return pb2.WordEmbeddingResponse(embeddings=embeddings, words=words)

    def get_embeddings(self, sentence):
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
        self.proc.kill()


@click.command()
@click.option('--log', help='log filepath')
@click.option('--debug', is_flag=True, help='debug')
def serve(log, debug):
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

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    fasttext_server = FasttextServer()
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
