#!/bin/bash
DOCKER_BINARY="docker"
IMAGE_NAME="llm_pq"
CONT_NAME='llm_pq-test'
DATA_PATH='/data' # mount the data directory to the container

MOUNT_PATH=$(<mount_path.txt)
echo "The value of MOUNT_PATH is: $MOUNT_PATH"
${DOCKER_BINARY} run --ipc host --gpus all  -v ${MOUNT_PATH}:/workspace/llm_pq -v ${DATA_PATH}:/data --name ${CONT_NAME} -it ${IMAGE_NAME}