ARG PYTHON_VERSION=3.7

ARG NAME_SPACE=shuzhi-amd64

FROM registry-vpc.cn-shanghai.aliyuncs.com/${NAME_SPACE}/horovod_docker_base:${PYTHON_VERSION} as builder

ENV PYPI_MIRROR "https://mirrors.aliyun.com/pypi/simple"

RUN pip config set global.index-url ${PYPI_MIRROR}

RUN pip install --upgrade pip

RUN pip install --no-cache-dir pyarmor

WORKDIR /build

COPY . /build

RUN bash tools/compress.sh

FROM registry-vpc.cn-shanghai.aliyuncs.com/${NAME_SPACE}/horovod_docker_base:${PYTHON_VERSION}

WORKDIR /yanqing

ENV PYPI_MIRROR "https://mirrors.aliyun.com/pypi/simple"

RUN pip config set global.index-url ${PYPI_MIRROR}

RUN pip install --upgrade pip

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get install graphviz -y

COPY --from=builder /build /yanqing

RUN pip install ./whl/cu100/torch-1.2.0-cp37-cp37m-manylinux1_x86_64.whl

RUN pip install ./whl/cu100/torchvision-0.4.0-cp37-cp37m-manylinux1_x86_64.whl

RUN rm -rf whl

RUN pip install -r requirements.txt

ENTRYPOINT [ "/sbin/my_init", "--" ]

CMD [ "bash" ]
