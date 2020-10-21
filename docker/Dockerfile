ARG PYTHON_VERSION

ARG NAME_SPACE

ARG DOCKER_BASE

FROM registry-vpc.cn-shanghai.aliyuncs.com/${NAME_SPACE}/${DOCKER_BASE}:${PYTHON_VERSION}

WORKDIR /pytorch

ENV PYPI_MIRROR "https://mirrors.aliyun.com/pypi/simple"

RUN pip config set global.index-url ${PYPI_MIRROR}

RUN pip install --upgrade pip

# RUN add-apt-repository ppa:deadsnakes/ppa

# RUN apt-get install graphviz -y

COPY . /pytorch

ARG REQUIREMENTS_FILE

ARG ENTRY_POINT

RUN pip install -r ${REQUIREMENTS_FILE}

ENTRYPOINT [ ENTRY_POINT, "--" ]

CMD [ "bash" ]
