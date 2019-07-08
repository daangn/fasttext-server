FROM python:3.7-slim

# fasttext python library
RUN apt-get -q update && apt-get -q install -y wget
RUN wget -q https://github.com/facebookresearch/fastText/archive/v0.9.1.tar.gz && \
      tar xfz v0.9.1.tar.gz && rm v0.9.1.tar.gz && mv fastText-0.9.1 fastText
RUN pip install --upgrade pip
RUN apt-get -q install -y build-essential
RUN cd fastText && pip install .

RUN pip install grpcio grpcio-tools
RUN pip install click

# for click library
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN pip install soyspacing gevent awscli

RUN mkdir -p /app
WORKDIR /app

ENTRYPOINT ["python"]
CMD ["server.py"]

HEALTHCHECK --interval=3s --timeout=2s \
  CMD ls /tmp/status || exit 1

COPY *.py /app/