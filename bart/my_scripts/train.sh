cuda=0
data=data/bin
save_dir=main

warmup_updates=400
lr=3e-05
dropout=0.1
update_freq=32
max_tokens=800
total_num_update=100000
bart=bart/bart.large/model.pt

CUDA_VISIBLE_DEVICES=$cuda python train.py $data \
    --restore-file $bart \
    --max-tokens $max_tokens \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout $dropout --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay \
    --lr $lr \
    --update-freq $update_freq \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --total-num-update $total_num_update  \
    --warmup-updates $warmup_updates \
    --no-epoch-checkpoints \
    --save-dir ckpt/$save_dir