FROM springtonyzhao/torch-lowprecision:latest
ARG PROJECT_DIR=/workspace
ADD . $PROJECT_DIR
WORKDIR $PROJECT_DIR
ENV HOME $PROJECT_DIR
# change data path and number of trainers here
ENV PYTHONPATH $PROJECT_DIR
RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 ca-certificates curl jq wget \
    git-lfs

RUN pip install deepspeed
    
CMD echo "===========END========="
CMD /bin/bash
