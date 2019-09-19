ARG PYTHON_VERSION=3.7

FROM registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/docker_base:${PYTHON_VERSION} as builder

ENV PYPI_MIRROR "https://mirrors.aliyun.com/pypi/simple"

RUN pip config set global.index-url ${PYPI_MIRROR}

RUN pip install --upgrade pip

RUN pip install --no-cache-dir pyarmor

WORKDIR /build

COPY . /build

RUN bash tools/compress.sh

FROM registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/docker_base:${PYTHON_VERSION}

WORKDIR /yanqing

ENV PYPI_MIRROR "https://mirrors.aliyun.com/pypi/simple"

RUN pip config set global.index-url ${PYPI_MIRROR}

RUN pip install --upgrade pip

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get install graphviz -y

COPY --from=builder /build /yanqing

RUN pip install -r requirements.txt

RUN pip install ./whl/cpu/torch-1.2.0+cpu-cp37-cp37m-manylinux1_x86_64.whl

RUN pip install ./whl/cpu/torchvision-0.4.0+cpu-cp37-cp37m-manylinux1_x86_64.whl

RUN rm -rf whl

ENTRYPOINT [ "/sbin/my_init", "--" ]

CMD [ "bash" ]
