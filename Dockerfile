FROM python:3.7-slim

# fasttext python library
RUN apt-get -q update && apt-get -q install -y wget
RUN apt-get install -y unzip
RUN pip install --upgrade pip
RUN apt-get -q install -y build-essential

#RUN wget -q https://github.com/facebookresearch/fastText/archive/v0.9.1.tar.gz && \
#      tar xfz v0.9.1.tar.gz && rm v0.9.1.tar.gz && mv fastText-0.9.1 fastText
#RUN cd fastText && pip install .

# for memory bug fix version
RUN wget https://github.com/facebookresearch/fastText/archive/40a77442a756ab160ae3465b26322f6e480405d9.zip -O fasttext.zip
RUN unzip fasttext.zip
RUN cd fastText-40a77442a756ab160ae3465b26322f6e480405d9 && pip install .

RUN pip install grpcio grpcio-tools
RUN pip install click

# for click library
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN pip install soyspacing gevent awscli

ENV GRPC_VERSION=1.23.0
RUN pip install grpcio==$GRPC_VERSION grpcio-tools==$GRPC_VERSION

RUN mkdir -p /app
WORKDIR /app

ENTRYPOINT ["python"]
CMD ["server.py"]

HEALTHCHECK --interval=3s --timeout=2s \
  CMD ls /tmp/status || exit 1

COPY *.py /app/