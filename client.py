from __future__ import print_function

import grpc

import fasttextserver_pb2 as pb2
import fasttextserver_pb2_grpc as pb2_grpc


def run():
  channel = grpc.insecure_channel('localhost:50051')
  stub = pb2_grpc.FasttextStub(channel)
  sentences = ['adf', '', 'xcvv w3r']

  request = pb2.ReloadRequest(model_type='word', filepath='models/skipgram.bin')
  response = stub.Reload(request)
  print("response: %s" % response.message)

  for sentence in sentences:
      request = pb2.SentenceRequest(sentence=sentence)
      response = stub.WordEmbedding(request)
      print("words: %s" % response.words)
      print("embeddings: %s" % response.embeddings)

  for sentence in sentences:
      request = pb2.SentenceRequest(sentence=sentence)
      response = stub.SentenceEmbedding(request)
      print("embeddings: %s" % response.embeddings)

  for sentence in sentences:
      request = pb2.SentenceRequest(sentence=sentence)
      response = stub.Predict(request)
      print("response: %s" % zip(response.labels, response.probs))

if __name__ == '__main__':
  run()
