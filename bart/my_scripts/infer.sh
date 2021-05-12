model_name=samsum
ckpt_dir=ckpt
out_name=samsum
CUDA_VISIBLE_DEVICES=0 python fairseq_cli/inference.py \
    --ckpt_dir ${ckpt_dir} \
    --ckpt_file ${model_name}.pt \
    --data_dir data/bin \
    --test_file data/test.source \
    --output_file summaries/${out_name}.txt \
    --batch_size 50 \
    --beam_size 4 \
    --min_len 5 \
    --max_len 100 \
    --block_ngram 3 \
    --len_penalty 0.5


# model_name=checkpoint_best
# ckpt_dir=ckpt/main
# out_name=main
# CUDA_VISIBLE_DEVICES=0 python fairseq_cli/inference.py \
#     --ckpt_dir ${ckpt_dir} \
#     --ckpt_file ${model_name}.pt \
#     --data_dir data/bin \
#     --test_file data/test.source \
#     --output_file summaries/${out_name}.txt \
#     --batch_size 10 \
#     --beam_size 4 \
#     --min_len 5 \
#     --max_len 100 \
#     --block_ngram 3 \
#     --len_penalty 0.5