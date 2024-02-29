gptq_folder="${ROOT_DIR}/3rd_party/gptq"
gptq_zs_folder="${gptq_folder}/zeroShot/"
cp bloom_rand.py ${gptq_zs_folder}/models/bloom.py
cp opt_rand.py ${gptq_zs_folder}/models/opt.py
cp utils.py ${gptq_zs_folder}/utils.py
cp lambada.py ${gptq_zs_folder}/tasks/local_datasets/lambada/lambada.py
cp gptq.py ${gptq_zs_folder}/models/gptq.py
# perplexity
cp opt_perplex.py ${gptq_folder}/opt.py
cp bloom_perplex.py ${gptq_folder}/bloom.py
cp adaqh_utils.py ${gptq_folder}/adaqh_utils.py
cp datautils.py ${gptq_folder}/datautils.py
