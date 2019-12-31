ARG PYTHON_VERSION=3.7

ARG NAME_SPACE=shuzhi-amd64

FROM registry-vpc.cn-shanghai.aliyuncs.com/${NAME_SPACE}/cuda_docker_base:${PYTHON_VERSION} as builder

ENV PYPI_MIRROR "https://mirrors.aliyun.com/pypi/simple"

RUN pip config set global.index-url ${PYPI_MIRROR}

RUN pip install --upgrade pip

RUN pip install --no-cache-dir pyarmor

WORKDIR /build

COPY . /build

RUN bash tools/compress.sh

FROM registry-vpc.cn-shanghai.aliyuncs.com/${NAME_SPACE}/cuda_docker_base:${PYTHON_VERSION}

WORKDIR /pytorch

ENV PYPI_MIRROR "https://mirrors.aliyun.com/pypi/simple"

RUN pip config set global.index-url ${PYPI_MIRROR}

RUN pip install --upgrade pip

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get install graphviz -y

COPY --from=builder /build /pytorch

RUN pip install torch==1.2.0 torchvision==0.4.0

RUN pip install -r requirements.txt

ENTRYPOINT [ "/usr/bin/dumb-init", "--" ]

CMD [ "bash" ]
