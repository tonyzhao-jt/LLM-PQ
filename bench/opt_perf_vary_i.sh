BATCH_SIZE=2
declare -A model_cards=(
    ["175b.s32"]="--h2 49152 --h1 12288 --b ${BATCH_SIZE} --i 32 --l 96 --unit GB --store"
    ["175b.s64"]="--h2 49152 --h1 12288 --b ${BATCH_SIZE} --i 64 --l 96 --unit GB --store"
    ["175b.s128"]="--h2 49152 --h1 12288 --b ${BATCH_SIZE} --i 128 --l 96 --unit GB --store"
    ["175b.s256"]="--h2 49152 --h1 12288 --b ${BATCH_SIZE} --i 256 --l 96 --unit GB --store"
    ["175b.s512"]="--h2 49152 --h1 12288 --b ${BATCH_SIZE} --i 512 --l 96 --unit GB --store"
)

models=("175b.s32" "175b.s64" "175b.s128" "175b.s256" "175b.s512")
# Loop through the models and run the Python script for each one
for model in "${models[@]}"; do
    echo "Running script for $model model..."
    python est_model_perfs.py ${model_cards[$model]} --model $model
done