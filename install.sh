pip3 install torch torchvision torchaudio
git clone --recursive https:///SpringWave1/QPipe.git
cd qpipe
cd 3rd_party/QLLM/3rd_party/LPTorch && pip install .
# cd ../transformers && git pull && pip install .
cd ../transformers && pip install .
cd ../.. && pip install -r requirements.txt
pip install .

# update gptq folder
cd scripts/accuracy/rand && bash update.sh
cd $ROOT_DIR
# update the gurobi license
cd scripts/gurobi_license && bash cp_gurobi.sh
cd $ROOT_DIR