from __future__ import print_function

import click
import grpc

import fasttextserver_pb2 as pb2
import fasttextserver_pb2_grpc as pb2_grpc

@click.group()
def cli():
    pass

@click.command()
@click.argument('sentence')
@click.option('--host', default='localhost:50051', help='host:port')
@click.option('-v', '--version', help='model version')
def word_emb(host, sentence, version):
    with grpc.insecure_channel(host) as channel:
        stub = pb2_grpc.FasttextStub(channel)
        request = pb2.SentenceRequest(sentence=sentence, version=version)
        response = stub.WordEmbedding(request)
        print("words: %s" % response.words)
        print("embeddings: %s" % response.embeddings)

@click.command()
@click.argument('sentences', nargs=-1)
@click.option('-h', '--host', default='localhost:50051', help='host:port')
@click.option('-v', '--version', help='model version')
def multi_word_emb(host, sentences, version):
    with grpc.insecure_channel(host) as channel:
        stub = pb2_grpc.FasttextStub(channel)
        request = pb2.MultiSentencesRequest(sentences=sentences, version=version)
        response = stub.MultiWordEmbeddings(request)
        for item in response.items:
            print("words: %s" % item.words)
            print("embeddings: %s" % item.embeddings)

@click.command()
@click.argument('model-type')
@click.option('--host', default='localhost:50051', help='host:port')
@click.option('-v', '--version', help='model version')
def restore(host, model_type, version):
    with grpc.insecure_channel(host) as channel:
        stub = pb2_grpc.FasttextStub(channel)
        request = pb2.ReloadRequest(model_type=model_type, version=version)
        response = stub.Reload(request)
        print("%s" % response.message)

@click.command()
@click.argument('sentence')
@click.option('--host', default='localhost:50051', help='host:port')
@click.option('-v', '--version', help='model version')
def sent_emb(host, sentence, version):
    with grpc.insecure_channel(host) as channel:
        stub = pb2_grpc.FasttextStub(channel)
        request = pb2.SentenceRequest(sentence=sentence, version=version)
        response = stub.SentenceEmbedding(request)
        print("embeddings: %s" % response.embeddings)

cli.add_command(word_emb)
cli.add_command(multi_word_emb)
cli.add_command(restore)
cli.add_command(sent_emb)

if __name__ == '__main__':
  cli()
