# PLM as an Annotator

This is the **Pytorch** code for our **ACL21** paper **Language Model as an Annotator: Exploring DialoGPT for Dialogue Summarization** [arXiv](https://arxiv.org/abs/2105.12544).

<p align="center">
  <img src="pic/main.png" width="750">
</p>

## Update
2021-08-02 release [DialoGPT annotator](https://github.com/xcfcode/PLM_annotator/tree/main/annotator).

## Outputs
Output summaries are available at [SAMSum](https://github.com/xcfcode/PLM_annotator/blob/main/bart/summaries/samsum.txt) and [AMI](https://github.com/xcfcode/PLM_annotator/blob/main/pgn/summaries/ami.txt).

## Codes

* For SAMSum dataset, please refer to the **[bart](https://github.com/xcfcode/PLM_annotator/tree/main/bart)** directory.
* For AMI dataset, please refer to the **[pgn](https://github.com/xcfcode/PLM_annotator/tree/main/pgn)** directory.

## Citation
```
@inproceedings{feng-etal-2021-language,
    title = "Language Model as an Annotator: Exploring {D}ialo{GPT} for Dialogue Summarization",
    author = "Feng, Xiachong  and
      Feng, Xiaocheng  and
      Qin, Libo  and
      Qin, Bing  and
      Liu, Ting",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.117",
    doi = "10.18653/v1/2021.acl-long.117",
    pages = "1479--1491",
    abstract = "Current dialogue summarization systems usually encode the text with a number of general semantic features (e.g., keywords and topics) to gain more powerful dialogue modeling capabilities. However, these features are obtained via open-domain toolkits that are dialog-agnostic or heavily relied on human annotations. In this paper, we show how DialoGPT, a pre-trained model for conversational response generation, can be developed as an unsupervised dialogue annotator, which takes advantage of dialogue background knowledge encoded in DialoGPT. We apply DialoGPT to label three types of features on two dialogue summarization datasets, SAMSum and AMI, and employ pre-trained and non pre-trained models as our summarizers. Experimental results show that our proposed method can obtain remarkable improvements on both datasets and achieves new state-of-the-art performance on the SAMSum dataset.",
}
```