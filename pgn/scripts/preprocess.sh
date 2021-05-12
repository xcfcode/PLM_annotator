data=data

python preprocess.py -train_src ${data}/train.txt.src \
     -train_tgt ${data}/train.txt.tgt \
     -valid_src ${data}/valid.txt.src \
     -valid_tgt ${data}/valid.txt.tgt \
     -shard_size 100 \
     -src_vocab_size 10000 \
     -tgt_vocab_size 10000 \
     -src_words_min_frequency 2 \
     -tgt_words_min_frequency 2 \
     -src_seq_length 15000 \
     -src_seq_length_trunc 11000 \
     -tgt_seq_length 700 \
     -tgt_seq_length_trunc 700 \
     -save_data ${data}/ami \
     -dynamic_dict \
     -share_vocab \
     -lower \
     -overwrite