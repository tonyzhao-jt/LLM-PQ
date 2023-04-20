pip3 install torch torchvision torchaudio
git clone --recursive https:///SpringWave1/QPipe.git
cd qpipe
cd 3rd_party/QLLM/3rd_party/LPTorch && pip install .
# cd ../transformers && git pull && pip install .
cd ../transformers && pip install .
cd ../.. && pip install -r requirements.txt
pip install .