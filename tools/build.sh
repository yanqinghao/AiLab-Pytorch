NAMESPACE=("shuzhi-amd64")
for i in ${NAMESPACE[*]}
do
    docker build --build-arg NAME_SPACE=${i} -t registry-vpc.cn-shanghai.aliyuncs.com/${i}/pytorch-docker-gpu:$1 -f docker/docker_yanqing_gpu/Dockerfile .
    docker build --build-arg NAME_SPACE=${i} -t registry-vpc.cn-shanghai.aliyuncs.com/${i}/pytorch-docker:$1 -f docker/docker_yanqing/Dockerfile .

    docker push registry-vpc.cn-shanghai.aliyuncs.com/${i}/pytorch-docker:$1
    docker push registry-vpc.cn-shanghai.aliyuncs.com/${i}/pytorch-docker-gpu:$1
done