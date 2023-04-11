declare -A model_cards=(
    ["125M"]="--h2 3072 --h1 768 --b 1 --i 4096 --l 24 --unit GB --store"
    ["350M"]="--h2 4096 --h1 1024 --b 1 --i 4096 --l 24 --unit GB --store"
    ["1.3b"]="--h2 8192 --h1 2048 --b 1 --i 4096 --l 24 --unit GB --store"
    ["2.7b"]="--h2 10240 --h1 2560 --b 1 --i 4096 --l 32 --unit GB --store"
    ["6.7b"]="--h2 16384 --h1 4096 --b 1 --i 4096 --l 32 --unit GB --store"
    ["13b"]="--h2 20480 --h1 5120 --b 1 --i 4096 --l 40 --unit GB --store"
    ["30b"]="--h2 28672 --h1 7168 --b 1 --i 4096 --l 48 --unit GB --store"
    ["66b"]="--h2 36864 --h1 9216 --b 1 --i 4096 --l 64 --unit GB --store"
    ["175b"]="--h2 49152 --h1 12288 --b 1 --i 4096 --l 96 --unit GB --store"
)

models=("125M" "350M" "1.3b" "2.7b" "6.7b" "13b" "30b" "66b" "175b")

# Loop through the models and run the Python script for each one
for model in "${models[@]}"; do
    echo "Running script for $model model..."
    python est_model_perfs.py ${model_cards[$model]} --model $model
done