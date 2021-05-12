# BART for SAMSum Dataset

This code is based on [Fairseq](https://github.com/pytorch/fairseq).

## Requirements
* We use Conda python 3.7 and strongly recommend that you create a new environment.
    * `conda create -n bart python=3.7`.
* Run the following command.
    * `pip install --editable ./`
    * `pip install -r requirements.txt`

## Data
You can get data [here](https://drive.google.com/drive/folders/1wLea1LdEv1jFQMtXr3bXJFiKvZnrGQLO?usp=sharing). Put them under the dir **data/\***.

## Reproduce Results
You can follow the following steps to reproduce the best results in our paper.

### download checkpoints
Download checkpoints [here](https://drive.google.com/drive/folders/1Osr3HXUPuGmh6-nCm8eSt_1ISJOaBhxy?usp=sharing). Put the checkpoint under the dir **ckpt/samsum.pt**.

### preprocess
* `sh ./my_scripts/bpe.sh`
* `sh ./my_scripts/binarize.sh`

### translate
* Produce final summaries.
    * `sh ./my_scripts/infer.sh`

### test rouge score
* `python py_rouge_test.py -c summaries/samsum.txt`

### ROUGE score
||ROUGE-1| ROUGE-2 | ROUGE-L |
| :---: | :---: | :---: | :---: |
| SAMSum | 53.70 | 28.79 | 50.81|

## From Scratch

### Download BART checkpoint
Download [bart.large](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz) checkpoint. Put it under the dir **bart/bart.large/\***.

### Preprocess
* `sh ./my_scripts/bpe.sh`
* `sh ./my_scripts/binarize.sh`

### Train
* `sh ./my_scripts/train.sh`

### Translate
Run the following command:
* `sh ./my_scripts/infer.sh`
   * set up **ckpt_dir** param first.
