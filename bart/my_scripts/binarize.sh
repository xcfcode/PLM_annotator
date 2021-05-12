data=data
destdir=data/bin

python fairseq_cli/preprocess.py \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref $data/train.bpe \
  --validpref $data/valid.bpe \
  --destdir $destdir \
  --workers 60 \
  --srcdict data/bpe/dict.txt \
  --tgtdict data/bpe/dict.txt