export CUDA_VISIBLE_DEVICES=3
model_storage_path='/data/llms/'
# Check if the LLM_PATH environmental variable is set
if [ -n "$LLM_PATH" ]; then
    model_storage_path="$LLM_PATH"
fi
model_name='opt'
model_size='66b'
if [ $# -eq 2 ]; then
    model_name="$1"
    model_size="$2"
fi

# Print the model name and size
echo "Model name: $model_name"
echo "Model size: $model_size"
export TRANSFORMERS_CACHE=$model_storage_path
python3 -c "import qllm; qllm.fast_download.from_pretrained_download('${model_name}', '${model_size}', '${model_storage_path}')"