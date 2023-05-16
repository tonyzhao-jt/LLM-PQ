model_storage_path='/data/llms/'
model_name='opt'
model_size='13b'
export TRANSFORMERS_CACHE=$model_storage_path
python3 -c "import qllm; qllm.fast_download.from_pretrained_download('${model_name}', '${model_size}', '${model_storage_path}')"