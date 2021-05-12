# Pointer-Generator for AMI Meeting Dataset
This code is based on [OpenNMT](https://github.com/OpenNMT/OpenNMT-py).

## Requirements
* We use Conda python 3.7 and strongly recommend that you create a new environment.
    * `conda create -n pgn python=3.7`.
* Run the following command.
    * `pip install -r requirements.txt`.

## Data
You can get data [here](https://drive.google.com/drive/folders/1VjuDhFxiv8t590-s_4HTX6BqOHhU89Ci?usp=sharing). Put them under the dir **data/\***.

## Reproduce Results
You can follow the following steps to reproduce the best results in our paper.

### download checkpoints
Download checkpoints [here](https://drive.google.com/drive/folders/1A9xjS_x1yhjwmtmOlyur16LCvOoOprwL?usp=sharing). Put the checkpoint under the dir **ckpt/ami.pt**.

### translate
* `sh ./scripts/infer.sh`

### test rouge score
* Change `pyrouge.Rouge155()` to your local path.
    * Output format `>> ROUGE(1/2/L): xx.xx-xx.xx-xx.xx`
* `python test_rouge.py -c summaries/ami.txt`

### ROUGE score
You will get following ROUGE scores.

||ROUGE-1| ROUGE-2 | ROUGE-L |
| :---: | :---: | :---: | :---: |
| AMI | 50.91 | 17.75 | 24.59 |

## From Scratch
### Preprocess
Run the following commands:
* `sh ./scripts/preprocess.sh`
* `sh ./scripts/embedding.sh`

### Train
Run the following command:
* `sh ./scripts/train.sh`

### Translate
Run the following command:
* `sh ./scripts/infer.sh`
   * set up **model_name** param first.