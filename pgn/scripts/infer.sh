cuda=0
model_name=ami
data_prefix=data
model=ckpt/${model_name}.pt
output=summaries/${model_name}.txt

CUDA_VISIBLE_DEVICES=${cuda} python translate.py -batch_size 1 \
               -src ${data_prefix}/test.txt.src \
               -tgt ${data_prefix}/test.txt.tgt \
               -beam_size 10 \
               -share_vocab \
               -dynamic_dict \
               -replace_unk \
               -model ${model} \
               -output ${output} \
               -block_ngram_repeat 3 \
               -gpu 0 \
               -min_length 280 \
               -max_length 450

sed -i 's/ <\/t>//g' ${output}
sed -i 's/<t> //g' ${output}