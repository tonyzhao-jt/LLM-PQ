FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
ARG PROJECT_DIR=/workspace
WORKDIR $PROJECT_DIR
# change data path and number of trainers here
RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 ca-certificates curl jq wget \
    git-lfs net-tools

RUN pip install gurobipy pulp sacrebleu
RUN pip install -U scikit-learn scipy matplotlib statsmodels
# ana
RUN pip install torch_tb_profiler


CMD echo "===========END========="
CMD /bin/bash