#!/bin/bash

set -e

VERSION=$(bash tools/version.sh)
DOCKERBASE=("suanpan-python-sdk", "suanpan-python-sdk-cuda")
if [[ $1 == "master" ]]; then
    echo "build from suanpan-python-sdk:3.7 and suanpan-python-sdk-cuda:3.7"
    echo "build pytorch-docker:latest and ${VERSION} and pytorch-docker-gpu:latest and ${VERSION}"
    TAGS=("3.7")
    BUILD_VERSIONS=(${VERSION})
    BUILD_TAGS=("latest")
else
    echo "build from suanpan-python-sdk:preview-3.7 and suanpan-python-sdk-cuda:preview-3.7"
    echo "build pytorch-docker:preview and preview-${VERSION} and pytorch-docker-gpu:preview and preview-${VERSION}"
    TAGS=("preview-3.7")
    BUILD_VERSIONS=("preview-${VERSION}")
    BUILD_TAGS=("preview")
fi
BUILDNAMES=("pytorch-docker", "pytorch-docker-gpu")
REQUIREMENTS=("requirements.txt", "requirements.txt")
ENTRYPOINT=("/sbin/my_init", "/usr/bin/dumb-init")
NAMESPACE="shuzhi-amd64"
for ((i = 0; i < ${#TAGS[@]}; i++)); do
    for ((j = 0; j < ${#BUILDNAMES[@]}; j++)); do
        docker build --build-arg NAME_SPACE=${NAMESPACE} --build-arg DOCKER_BASE=${DOCKERBASE[j]} \
            --build-arg PYTHON_VERSION=${TAGS[i]} --build-arg REQUIREMENTS_FILE=${REQUIREMENTS[j]} \
            --build-arg ENTRY_POINT=${ENTRYPOINT[j]} -t \
            registry-vpc.cn-shanghai.aliyuncs.com/${NAMESPACE}/${BUILDNAMES[j]}:${BUILD_VERSIONS[i]} \
            -f docker/Dockerfile .
        docker push registry-vpc.cn-shanghai.aliyuncs.com/${NAMESPACE}/${BUILDNAMES[j]}:${BUILD_VERSIONS[i]}

        docker tag registry-vpc.cn-shanghai.aliyuncs.com/${NAMESPACE}/${BUILDNAMES[j]}:${BUILD_VERSIONS[i]} \
            registry-vpc.cn-shanghai.aliyuncs.com/${NAMESPACE}/${BUILDNAMES[j]}:${BUILD_TAGS[i]}
        docker push registry-vpc.cn-shanghai.aliyuncs.com/${NAMESPACE}/${BUILDNAMES[j]}:${BUILD_TAGS[i]}
    done
done
