from __future__ import print_function

import grpc

import fasttextserver_pb2 as pb2
import fasttextserver_pb2_grpc as pb2_grpc


def run():
  channel = grpc.insecure_channel('localhost:50051')
  stub = pb2_grpc.FasttextStub(channel)
  sentences = ['adf', '', 'xcvv w3r']
  for sentence in sentences:
      request = pb2.WordEmbeddingRequest(sentence=sentence)
      response = stub.WordEmbedding(request)
      print("words: %s" % response.words)
      print("embeddings: %s" % response.embeddings)


if __name__ == '__main__':
  run()
