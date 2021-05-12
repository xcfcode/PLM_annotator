# -*- coding: utf-8 -*-
# @Time    : 2020/10/13 12:23 下午
# @Author  : Xiachong Feng
# @File    : inference.py
# @Software: PyCharm
import argparse

import torch
from fairseq.models.bart import BARTModel
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference For SAMSum")
    parser.add_argument("--ckpt_dir", default="checkpoints/", help="")
    parser.add_argument("--ckpt_file", default="checkpoint_best.pt", help="")
    parser.add_argument("--data_dir", default="bin", help="")
    parser.add_argument("--test_file", default="test.source", help="test dialogue path")
    parser.add_argument("--output_file", default="", help="output file path")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--beam_size", default=5, type=int, help="beam width")
    parser.add_argument("--min_len", default=15, type=int, help="min length of generated summary")
    parser.add_argument("--max_len", default=140, type=int, help="max length of generated summary")
    parser.add_argument("--block_ngram", default=3, type=int, help="no repeat n-gram")
    parser.add_argument("--len_penalty", default=2.0, type=float, help="length penalty")
    args = parser.parse_args()

    # load model
    print(time.strftime("Start at : %Y-%m-%d %H:%M:%S", time.localtime()))
    bart = BARTModel.from_pretrained(
        args.ckpt_dir,
        checkpoint_file=args.ckpt_file,
        data_name_or_path=args.data_dir
    )
    print(time.strftime("End at : %Y-%m-%d %H:%M:%S", time.localtime()))

    bart.cuda()
    bart.eval()
    bart.half()
    count = 1
    bsz = args.batch_size
    with open(args.test_file, encoding="utf-8") as source, open(args.output_file, 'w', encoding="utf-8") as fout:
        sline = source.readline().strip()
        slines = [sline]
        for sline in source:
            if count % bsz == 0:
                with torch.no_grad():
                    hypotheses_batch = bart.sample(slines, beam=args.beam_size, lenpen=args.len_penalty,
                                                   max_len_b=args.max_len,
                                                   min_len=args.min_len,
                                                   no_repeat_ngram_size=args.block_ngram)

                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + '\n')
                    fout.flush()
                slines = []

            slines.append(sline.strip())
            count += 1
        if slines != []:
            hypotheses_batch = bart.sample(slines, beam=args.beam_size, lenpen=args.len_penalty, max_len_b=args.max_len,
                                           min_len=args.min_len, no_repeat_ngram_size=args.block_ngram)
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + '\n')
                fout.flush()
