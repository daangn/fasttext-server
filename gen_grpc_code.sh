#!/bin/bash
source config.sh
PROTO_FILEPATH=fasttextserver.proto
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. $PROTO_FILEPATH
grpc_tools_ruby_protoc -I. --ruby_out=$OUTPUT_PATH --grpc_out=$OUTPUT_PATH $PROTO_FILEPATH