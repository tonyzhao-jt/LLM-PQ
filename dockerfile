FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
ARG PROJECT_DIR=/workspace
WORKDIR $PROJECT_DIR
# change data path and number of trainers here
RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 ca-certificates curl jq wget \
    git-lfs net-tools

RUN python3 -m pip install gurobipy pulp sacrebleu
RUN python3 -m pip install -U scikit-learn scipy matplotlib statsmodels
# ana
RUN python3 -m pip install torch_tb_profiler
# dot env
RUN python3 -m pip install python-dotenv
# logger
RUN python3 -m pip install colorlog

ENV ROOT_DIR=/workspace/shaq

CMD echo "===========END========="
CMD /bin/bash