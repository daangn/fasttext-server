# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: fasttextserver.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='fasttextserver.proto',
  package='nlp',
  syntax='proto3',
  serialized_pb=_b('\n\x14\x66\x61sttextserver.proto\x12\x03nlp\"E\n\x0fSentenceRequest\x12\x10\n\x08sentence\x18\x01 \x01(\t\x12\x0f\n\x07version\x18\x02 \x01(\t\x12\x0f\n\x07spacing\x18\x03 \x01(\x08\"L\n\x15MultiSentencesRequest\x12\x11\n\tsentences\x18\x01 \x03(\t\x12\x0f\n\x07version\x18\x02 \x01(\t\x12\x0f\n\x07spacing\x18\x03 \x01(\x08\":\n\x15WordEmbeddingResponse\x12\x12\n\nembeddings\x18\x01 \x03(\x02\x12\r\n\x05words\x18\x02 \x03(\t\"H\n\x1bMultiWordEmbeddingsResponse\x12)\n\x05items\x18\x01 \x03(\x0b\x32\x1a.nlp.WordEmbeddingResponse\"/\n\x19SentenceEmbeddingResponse\x12\x12\n\nembeddings\x18\x01 \x03(\x02\"4\n\rReloadRequest\x12\x12\n\nmodel_type\x18\x01 \x01(\t\x12\x0f\n\x07version\x18\x02 \x01(\t\"\x1b\n\x08Response\x12\x0f\n\x07message\x18\x01 \x01(\t\"0\n\x0fPredictResponse\x12\x0e\n\x06labels\x18\x01 \x03(\t\x12\r\n\x05probs\x18\x02 \x03(\x02\x32\xdb\x02\n\x08\x46\x61sttext\x12\x43\n\rWordEmbedding\x12\x14.nlp.SentenceRequest\x1a\x1a.nlp.WordEmbeddingResponse\"\x00\x12U\n\x13MultiWordEmbeddings\x12\x1a.nlp.MultiSentencesRequest\x1a .nlp.MultiWordEmbeddingsResponse\"\x00\x12K\n\x11SentenceEmbedding\x12\x14.nlp.SentenceRequest\x1a\x1e.nlp.SentenceEmbeddingResponse\"\x00\x12\x37\n\x07Predict\x12\x14.nlp.SentenceRequest\x1a\x14.nlp.PredictResponse\"\x00\x12-\n\x06Reload\x12\x12.nlp.ReloadRequest\x1a\r.nlp.Response\"\x00\x62\x06proto3')
)




_SENTENCEREQUEST = _descriptor.Descriptor(
  name='SentenceRequest',
  full_name='nlp.SentenceRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='sentence', full_name='nlp.SentenceRequest.sentence', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='version', full_name='nlp.SentenceRequest.version', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='spacing', full_name='nlp.SentenceRequest.spacing', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=29,
  serialized_end=98,
)


_MULTISENTENCESREQUEST = _descriptor.Descriptor(
  name='MultiSentencesRequest',
  full_name='nlp.MultiSentencesRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='sentences', full_name='nlp.MultiSentencesRequest.sentences', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='version', full_name='nlp.MultiSentencesRequest.version', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='spacing', full_name='nlp.MultiSentencesRequest.spacing', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=100,
  serialized_end=176,
)


_WORDEMBEDDINGRESPONSE = _descriptor.Descriptor(
  name='WordEmbeddingResponse',
  full_name='nlp.WordEmbeddingResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='embeddings', full_name='nlp.WordEmbeddingResponse.embeddings', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='words', full_name='nlp.WordEmbeddingResponse.words', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=178,
  serialized_end=236,
)


_MULTIWORDEMBEDDINGSRESPONSE = _descriptor.Descriptor(
  name='MultiWordEmbeddingsResponse',
  full_name='nlp.MultiWordEmbeddingsResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='items', full_name='nlp.MultiWordEmbeddingsResponse.items', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=238,
  serialized_end=310,
)


_SENTENCEEMBEDDINGRESPONSE = _descriptor.Descriptor(
  name='SentenceEmbeddingResponse',
  full_name='nlp.SentenceEmbeddingResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='embeddings', full_name='nlp.SentenceEmbeddingResponse.embeddings', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=312,
  serialized_end=359,
)


_RELOADREQUEST = _descriptor.Descriptor(
  name='ReloadRequest',
  full_name='nlp.ReloadRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='model_type', full_name='nlp.ReloadRequest.model_type', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='version', full_name='nlp.ReloadRequest.version', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=361,
  serialized_end=413,
)


_RESPONSE = _descriptor.Descriptor(
  name='Response',
  full_name='nlp.Response',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='message', full_name='nlp.Response.message', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=415,
  serialized_end=442,
)


_PREDICTRESPONSE = _descriptor.Descriptor(
  name='PredictResponse',
  full_name='nlp.PredictResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='labels', full_name='nlp.PredictResponse.labels', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='probs', full_name='nlp.PredictResponse.probs', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=444,
  serialized_end=492,
)

_MULTIWORDEMBEDDINGSRESPONSE.fields_by_name['items'].message_type = _WORDEMBEDDINGRESPONSE
DESCRIPTOR.message_types_by_name['SentenceRequest'] = _SENTENCEREQUEST
DESCRIPTOR.message_types_by_name['MultiSentencesRequest'] = _MULTISENTENCESREQUEST
DESCRIPTOR.message_types_by_name['WordEmbeddingResponse'] = _WORDEMBEDDINGRESPONSE
DESCRIPTOR.message_types_by_name['MultiWordEmbeddingsResponse'] = _MULTIWORDEMBEDDINGSRESPONSE
DESCRIPTOR.message_types_by_name['SentenceEmbeddingResponse'] = _SENTENCEEMBEDDINGRESPONSE
DESCRIPTOR.message_types_by_name['ReloadRequest'] = _RELOADREQUEST
DESCRIPTOR.message_types_by_name['Response'] = _RESPONSE
DESCRIPTOR.message_types_by_name['PredictResponse'] = _PREDICTRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SentenceRequest = _reflection.GeneratedProtocolMessageType('SentenceRequest', (_message.Message,), dict(
  DESCRIPTOR = _SENTENCEREQUEST,
  __module__ = 'fasttextserver_pb2'
  # @@protoc_insertion_point(class_scope:nlp.SentenceRequest)
  ))
_sym_db.RegisterMessage(SentenceRequest)

MultiSentencesRequest = _reflection.GeneratedProtocolMessageType('MultiSentencesRequest', (_message.Message,), dict(
  DESCRIPTOR = _MULTISENTENCESREQUEST,
  __module__ = 'fasttextserver_pb2'
  # @@protoc_insertion_point(class_scope:nlp.MultiSentencesRequest)
  ))
_sym_db.RegisterMessage(MultiSentencesRequest)

WordEmbeddingResponse = _reflection.GeneratedProtocolMessageType('WordEmbeddingResponse', (_message.Message,), dict(
  DESCRIPTOR = _WORDEMBEDDINGRESPONSE,
  __module__ = 'fasttextserver_pb2'
  # @@protoc_insertion_point(class_scope:nlp.WordEmbeddingResponse)
  ))
_sym_db.RegisterMessage(WordEmbeddingResponse)

MultiWordEmbeddingsResponse = _reflection.GeneratedProtocolMessageType('MultiWordEmbeddingsResponse', (_message.Message,), dict(
  DESCRIPTOR = _MULTIWORDEMBEDDINGSRESPONSE,
  __module__ = 'fasttextserver_pb2'
  # @@protoc_insertion_point(class_scope:nlp.MultiWordEmbeddingsResponse)
  ))
_sym_db.RegisterMessage(MultiWordEmbeddingsResponse)

SentenceEmbeddingResponse = _reflection.GeneratedProtocolMessageType('SentenceEmbeddingResponse', (_message.Message,), dict(
  DESCRIPTOR = _SENTENCEEMBEDDINGRESPONSE,
  __module__ = 'fasttextserver_pb2'
  # @@protoc_insertion_point(class_scope:nlp.SentenceEmbeddingResponse)
  ))
_sym_db.RegisterMessage(SentenceEmbeddingResponse)

ReloadRequest = _reflection.GeneratedProtocolMessageType('ReloadRequest', (_message.Message,), dict(
  DESCRIPTOR = _RELOADREQUEST,
  __module__ = 'fasttextserver_pb2'
  # @@protoc_insertion_point(class_scope:nlp.ReloadRequest)
  ))
_sym_db.RegisterMessage(ReloadRequest)

Response = _reflection.GeneratedProtocolMessageType('Response', (_message.Message,), dict(
  DESCRIPTOR = _RESPONSE,
  __module__ = 'fasttextserver_pb2'
  # @@protoc_insertion_point(class_scope:nlp.Response)
  ))
_sym_db.RegisterMessage(Response)

PredictResponse = _reflection.GeneratedProtocolMessageType('PredictResponse', (_message.Message,), dict(
  DESCRIPTOR = _PREDICTRESPONSE,
  __module__ = 'fasttextserver_pb2'
  # @@protoc_insertion_point(class_scope:nlp.PredictResponse)
  ))
_sym_db.RegisterMessage(PredictResponse)


try:
  # THESE ELEMENTS WILL BE DEPRECATED.
  # Please use the generated *_pb2_grpc.py files instead.
  import grpc
  from grpc.beta import implementations as beta_implementations
  from grpc.beta import interfaces as beta_interfaces
  from grpc.framework.common import cardinality
  from grpc.framework.interfaces.face import utilities as face_utilities


  class FasttextStub(object):
    # missing associated documentation comment in .proto file
    pass

    def __init__(self, channel):
      """Constructor.

      Args:
        channel: A grpc.Channel.
      """
      self.WordEmbedding = channel.unary_unary(
          '/nlp.Fasttext/WordEmbedding',
          request_serializer=SentenceRequest.SerializeToString,
          response_deserializer=WordEmbeddingResponse.FromString,
          )
      self.MultiWordEmbeddings = channel.unary_unary(
          '/nlp.Fasttext/MultiWordEmbeddings',
          request_serializer=MultiSentencesRequest.SerializeToString,
          response_deserializer=MultiWordEmbeddingsResponse.FromString,
          )
      self.SentenceEmbedding = channel.unary_unary(
          '/nlp.Fasttext/SentenceEmbedding',
          request_serializer=SentenceRequest.SerializeToString,
          response_deserializer=SentenceEmbeddingResponse.FromString,
          )
      self.Predict = channel.unary_unary(
          '/nlp.Fasttext/Predict',
          request_serializer=SentenceRequest.SerializeToString,
          response_deserializer=PredictResponse.FromString,
          )
      self.Reload = channel.unary_unary(
          '/nlp.Fasttext/Reload',
          request_serializer=ReloadRequest.SerializeToString,
          response_deserializer=Response.FromString,
          )


  class FasttextServicer(object):
    # missing associated documentation comment in .proto file
    pass

    def WordEmbedding(self, request, context):
      # missing associated documentation comment in .proto file
      pass
      context.set_code(grpc.StatusCode.UNIMPLEMENTED)
      context.set_details('Method not implemented!')
      raise NotImplementedError('Method not implemented!')

    def MultiWordEmbeddings(self, request, context):
      # missing associated documentation comment in .proto file
      pass
      context.set_code(grpc.StatusCode.UNIMPLEMENTED)
      context.set_details('Method not implemented!')
      raise NotImplementedError('Method not implemented!')

    def SentenceEmbedding(self, request, context):
      # missing associated documentation comment in .proto file
      pass
      context.set_code(grpc.StatusCode.UNIMPLEMENTED)
      context.set_details('Method not implemented!')
      raise NotImplementedError('Method not implemented!')

    def Predict(self, request, context):
      # missing associated documentation comment in .proto file
      pass
      context.set_code(grpc.StatusCode.UNIMPLEMENTED)
      context.set_details('Method not implemented!')
      raise NotImplementedError('Method not implemented!')

    def Reload(self, request, context):
      # missing associated documentation comment in .proto file
      pass
      context.set_code(grpc.StatusCode.UNIMPLEMENTED)
      context.set_details('Method not implemented!')
      raise NotImplementedError('Method not implemented!')


  def add_FasttextServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'WordEmbedding': grpc.unary_unary_rpc_method_handler(
            servicer.WordEmbedding,
            request_deserializer=SentenceRequest.FromString,
            response_serializer=WordEmbeddingResponse.SerializeToString,
        ),
        'MultiWordEmbeddings': grpc.unary_unary_rpc_method_handler(
            servicer.MultiWordEmbeddings,
            request_deserializer=MultiSentencesRequest.FromString,
            response_serializer=MultiWordEmbeddingsResponse.SerializeToString,
        ),
        'SentenceEmbedding': grpc.unary_unary_rpc_method_handler(
            servicer.SentenceEmbedding,
            request_deserializer=SentenceRequest.FromString,
            response_serializer=SentenceEmbeddingResponse.SerializeToString,
        ),
        'Predict': grpc.unary_unary_rpc_method_handler(
            servicer.Predict,
            request_deserializer=SentenceRequest.FromString,
            response_serializer=PredictResponse.SerializeToString,
        ),
        'Reload': grpc.unary_unary_rpc_method_handler(
            servicer.Reload,
            request_deserializer=ReloadRequest.FromString,
            response_serializer=Response.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'nlp.Fasttext', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


  class BetaFasttextServicer(object):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This class was generated
    only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0."""
    # missing associated documentation comment in .proto file
    pass
    def WordEmbedding(self, request, context):
      # missing associated documentation comment in .proto file
      pass
      context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)
    def MultiWordEmbeddings(self, request, context):
      # missing associated documentation comment in .proto file
      pass
      context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)
    def SentenceEmbedding(self, request, context):
      # missing associated documentation comment in .proto file
      pass
      context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)
    def Predict(self, request, context):
      # missing associated documentation comment in .proto file
      pass
      context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)
    def Reload(self, request, context):
      # missing associated documentation comment in .proto file
      pass
      context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)


  class BetaFasttextStub(object):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This class was generated
    only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0."""
    # missing associated documentation comment in .proto file
    pass
    def WordEmbedding(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
      # missing associated documentation comment in .proto file
      pass
      raise NotImplementedError()
    WordEmbedding.future = None
    def MultiWordEmbeddings(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
      # missing associated documentation comment in .proto file
      pass
      raise NotImplementedError()
    MultiWordEmbeddings.future = None
    def SentenceEmbedding(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
      # missing associated documentation comment in .proto file
      pass
      raise NotImplementedError()
    SentenceEmbedding.future = None
    def Predict(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
      # missing associated documentation comment in .proto file
      pass
      raise NotImplementedError()
    Predict.future = None
    def Reload(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
      # missing associated documentation comment in .proto file
      pass
      raise NotImplementedError()
    Reload.future = None


  def beta_create_Fasttext_server(servicer, pool=None, pool_size=None, default_timeout=None, maximum_timeout=None):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This function was
    generated only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0"""
    request_deserializers = {
      ('nlp.Fasttext', 'MultiWordEmbeddings'): MultiSentencesRequest.FromString,
      ('nlp.Fasttext', 'Predict'): SentenceRequest.FromString,
      ('nlp.Fasttext', 'Reload'): ReloadRequest.FromString,
      ('nlp.Fasttext', 'SentenceEmbedding'): SentenceRequest.FromString,
      ('nlp.Fasttext', 'WordEmbedding'): SentenceRequest.FromString,
    }
    response_serializers = {
      ('nlp.Fasttext', 'MultiWordEmbeddings'): MultiWordEmbeddingsResponse.SerializeToString,
      ('nlp.Fasttext', 'Predict'): PredictResponse.SerializeToString,
      ('nlp.Fasttext', 'Reload'): Response.SerializeToString,
      ('nlp.Fasttext', 'SentenceEmbedding'): SentenceEmbeddingResponse.SerializeToString,
      ('nlp.Fasttext', 'WordEmbedding'): WordEmbeddingResponse.SerializeToString,
    }
    method_implementations = {
      ('nlp.Fasttext', 'MultiWordEmbeddings'): face_utilities.unary_unary_inline(servicer.MultiWordEmbeddings),
      ('nlp.Fasttext', 'Predict'): face_utilities.unary_unary_inline(servicer.Predict),
      ('nlp.Fasttext', 'Reload'): face_utilities.unary_unary_inline(servicer.Reload),
      ('nlp.Fasttext', 'SentenceEmbedding'): face_utilities.unary_unary_inline(servicer.SentenceEmbedding),
      ('nlp.Fasttext', 'WordEmbedding'): face_utilities.unary_unary_inline(servicer.WordEmbedding),
    }
    server_options = beta_implementations.server_options(request_deserializers=request_deserializers, response_serializers=response_serializers, thread_pool=pool, thread_pool_size=pool_size, default_timeout=default_timeout, maximum_timeout=maximum_timeout)
    return beta_implementations.server(method_implementations, options=server_options)


  def beta_create_Fasttext_stub(channel, host=None, metadata_transformer=None, pool=None, pool_size=None):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This function was
    generated only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0"""
    request_serializers = {
      ('nlp.Fasttext', 'MultiWordEmbeddings'): MultiSentencesRequest.SerializeToString,
      ('nlp.Fasttext', 'Predict'): SentenceRequest.SerializeToString,
      ('nlp.Fasttext', 'Reload'): ReloadRequest.SerializeToString,
      ('nlp.Fasttext', 'SentenceEmbedding'): SentenceRequest.SerializeToString,
      ('nlp.Fasttext', 'WordEmbedding'): SentenceRequest.SerializeToString,
    }
    response_deserializers = {
      ('nlp.Fasttext', 'MultiWordEmbeddings'): MultiWordEmbeddingsResponse.FromString,
      ('nlp.Fasttext', 'Predict'): PredictResponse.FromString,
      ('nlp.Fasttext', 'Reload'): Response.FromString,
      ('nlp.Fasttext', 'SentenceEmbedding'): SentenceEmbeddingResponse.FromString,
      ('nlp.Fasttext', 'WordEmbedding'): WordEmbeddingResponse.FromString,
    }
    cardinalities = {
      'MultiWordEmbeddings': cardinality.Cardinality.UNARY_UNARY,
      'Predict': cardinality.Cardinality.UNARY_UNARY,
      'Reload': cardinality.Cardinality.UNARY_UNARY,
      'SentenceEmbedding': cardinality.Cardinality.UNARY_UNARY,
      'WordEmbedding': cardinality.Cardinality.UNARY_UNARY,
    }
    stub_options = beta_implementations.stub_options(host=host, metadata_transformer=metadata_transformer, request_serializers=request_serializers, response_deserializers=response_deserializers, thread_pool=pool, thread_pool_size=pool_size)
    return beta_implementations.dynamic_stub(channel, 'nlp.Fasttext', cardinalities, options=stub_options)
except ImportError:
  pass
# @@protoc_insertion_point(module_scope)
