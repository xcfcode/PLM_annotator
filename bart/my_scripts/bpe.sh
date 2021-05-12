DIR=data
for SPLIT in train valid
do
  for LANG in source target
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json data/bpe/encoder.json \
    --vocab-bpe data/bpe/vocab.bpe \
    --inputs "$DIR/$SPLIT.$LANG" \
    --outputs "$DIR/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done