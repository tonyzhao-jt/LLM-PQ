gptq_zs_folder="/workspace/qpipe/3rd_party/gptq/zeroShot/"
cp bloom_rand.py ${gptq_zs_folder}/models/bloom.py
cp opt_rand.py ${gptq_zs_folder}/models/opt.py
cp utils.py ${gptq_zs_folder}/utils.py
cp lambada.py ${gptq_zs_folder}/tasks/local_datasets/lambada/lambada.py
cp gptq.py ${gptq_zs_folder}/models/gptq.py