cuda=0
name=ami
data_prefix=data
data=${data_prefix}/ami

CUDA_VISIBLE_DEVICES=${cuda} python train.py -save_model ckpt/${name} \
           -data ${data} \
           -batch_size 1 \
           -learning_rate 0.001 \
           -share_embeddings \
           -pre_word_vecs_enc ${data_prefix}/embeddings.enc.pt \
           -pre_word_vecs_dec ${data_prefix}/embeddings.dec.pt \
           -save_checkpoint_steps 100 \
           -seed 777 \
           -optim adam \
           -max_grad_norm 2 \
           -report_every 100 \
           -word_vec_size 300 \
           -encoder_type rnn \
           -rnn_size 200 \
           -gpu_ranks 0 \
           -valid_steps 100 \
           -copy_attn \
           -reuse_copy_attn \
           -log_file logs/${name}.txt \
           -save_config logs/${name}.txt


CUDA_VISIBLE_DEVICES=${cuda} python train.py -save_model ckpt/${name} \
           -data ${data} \
           -batch_size 1 \
           -learning_rate 0.001 \
           -share_embeddings \
           -pre_word_vecs_enc ${data_prefix}/embeddings.enc.pt \
           -pre_word_vecs_dec ${data_prefix}/embeddings.dec.pt \
           -save_checkpoint_steps 100 \
           -seed 777 \
           -optim adam \
           -max_grad_norm 2 \
           -report_every 100 \
           -word_vec_size 300 \
           -encoder_type rnn \
           -rnn_size 200 \
           -gpu_ranks 0 \
           -valid_steps 100 \
           -copy_attn \
           -reuse_copy_attn \
           -log_file logs/${name}.txt