NAMESPACE=(shuzhi,shuzhi-amd64)
for i in ${NAMESPACE}
do
    docker build --build-arg NAME_SPACE=${i} -t registry-vpc.cn-shanghai.aliyuncs.com/${i}/pytorch-docker-gpu:$1 -f docker/docker_yanqing_gpu/Dockerfile .
    docker build --build-arg NAME_SPACE=${i} -t registry-vpc.cn-shanghai.aliyuncs.com/${i}/pytorch-stream-gpu:$1 -f docker/stream_yanqing_gpu/Dockerfile .
    sed -i "s/cpu/cu100/g" .dockerignore
    docker build --build-arg NAME_SPACE=${i} -t registry-vpc.cn-shanghai.aliyuncs.com/${i}/pytorch-docker:$1 -f docker/docker_yanqing/Dockerfile .
    docker build --build-arg NAME_SPACE=${i} -t registry-vpc.cn-shanghai.aliyuncs.com/${i}/pytorch-stream:$1 -f docker/stream_yanqing/Dockerfile .
    sed -i "s/cu100/cpu/g" .dockerignore

    docker push registry-vpc.cn-shanghai.aliyuncs.com/${i}/pytorch-docker:$1
    docker push registry-vpc.cn-shanghai.aliyuncs.com/${i}/pytorch-stream:$1
    docker push registry-vpc.cn-shanghai.aliyuncs.com/${i}/pytorch-docker-gpu:$1
    docker push registry-vpc.cn-shanghai.aliyuncs.com/${i}/pytorch-stream-gpu:$1
done