FROM daangn/fasttext

RUN apt-get update && apt-get install python3 -y

RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && rm get-pip.py
RUN python3 -m pip install --upgrade pip

# grpc
ENV GRPC_PYTHON_VERSION 1.6.0
RUN pip3 install grpcio==${GRPC_PYTHON_VERSION} grpcio-tools==${GRPC_PYTHON_VERSION}
RUN pip3 install click

# for click library
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y python3-dev cmake gcc
RUN cd fastText && pip3 install .

RUN pip3 install soyspacing
RUN pip3 install gevent
RUN pip3 install awscli

RUN mkdir -p /app
WORKDIR /app

ENTRYPOINT ["python3"]
CMD ["server.py"]

HEALTHCHECK --interval=3s --timeout=2s \
  CMD ls /tmp/status || exit 1

COPY *.py /app/