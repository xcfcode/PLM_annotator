dict_file=data/ami.vocab.pt
output_file=data/embeddings

python embeddings_to_torch.py -emb_file_both data/glove.6B.300d.txt \
    -dict_file ${dict_file} \
    -output_file ${output_file}