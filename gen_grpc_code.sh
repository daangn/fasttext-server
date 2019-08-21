#!/bin/bash
source config.sh
PROTO_FILEPATH=fasttextserver.proto
docker run --rm -it -v "$(pwd)":/app \
  daangn/fasttext-server -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. $PROTO_FILEPATH

# if no grpc_tools_ruby_protoc, gem install grpc-tools
grpc_tools_ruby_protoc -I. --ruby_out=$OUTPUT_PATH --grpc_out=$OUTPUT_PATH $PROTO_FILEPATH