# DialoGPT Annotator

This code mainly shows how we annotate a dialogue from AMI and SAMSum using [DialoGPT](https://arxiv.org/abs/1911.00536).

## Requirements
* `conda create -n tfs python=3.7`
* `pip install -r requirements.txt`

## Get loss
Firstly, run the following command, you will get a dir ***loss/samsum/bpe*** or ***loss/ami/bpe*** that stores three files: ***train_loss.json***, ***valid_loss.json*** and ***test_loss.json***.
* For SAMSum: `python get_loss.py -d samsum`
* For AMI: `python get_loss.py -d ami`

Secondly, we recover word-level loss, you will get a dir ***loss/samsum/word*** or ***loss/ami/word***
> Note that DialoGPT uses BPE to tokenize texts, thus, losses are calculated at the sub-word level. We recover the word-level predicted loss by averaging the losses of multiple sub-words.
* For SAMSum: `python recover_word_loss.py -d samsum`
* For AMI: `python recover_word_loss.py -d ami`

## Get dialogue context representation
Run following commands, you will get a dir ***rep/samsum*** or ***rep/ami*** that stores three files: ***train_rep.json***, ***valid_rep.json*** and ***test_rep.json***.
* For SAMSum: `python get_representation_samsum.py`
* For AMI: `python get_representation_ami.py`

## Calculate cosine similarity
Run following commands, you will get a dir ***rep/samsum/sim*** or ***rep/samsum/sim*** that stores three files: ***train_sim.json***, ***valid_sim.json*** and ***test_sim.json***.
* For SAMSum: `python cosine_sim.py -d samsum`
* For AMI: `python cosine_sim.py -d ami`

## Annotate
Run following commands, you will get a dir ***data/samsum/final*** or ***data/ami/final*** that stores final output files.
* For SAMSum: `python annotate.py -d samsum`
* For AMI: `python annotate.py -d ami`