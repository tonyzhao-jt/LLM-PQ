DOCKER_BINARY="docker"
IMAGE_NAME='llm_pq'
DOCKER_BUILDKIT=1 ${DOCKER_BINARY} build -f Dockerfile -t ${IMAGE_NAME}:latest .