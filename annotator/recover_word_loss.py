# -*- coding: utf-8 -*-
import argparse
import json
import os

from utils import load_json


def get_datas(train_path, valid_path, test_path):
    train_datas = load_json(train_path)
    valid_datas = load_json(valid_path)
    test_datas = load_json(test_path)
    return train_datas, valid_datas, test_datas


def convert_loss2pseudo_bpe(subwords, losses):
    res = []
    for subword, loss in zip(subwords, losses):
        loss = str(loss)
        loss = loss + "$"
        if "Ġ" in subword:
            loss = "Ġ" + loss
        if "<|endoftext|>" in subword:
            loss = "Ġ" + loss + "Ġ"
        res.append(loss)
    assert len(res) == len(subwords)
    return res


def average_loss(word_level_losses):
    res = []
    losses = word_level_losses.split()
    for loss in losses:
        micro_losses = loss.split("$")[:-1]
        avg_loss = sum([float(micro_loss) for micro_loss in micro_losses]) / len(micro_losses)
        res.append(avg_loss)
    return res


def recover_word_level(subwords, losses):
    losses = convert_loss2pseudo_bpe(subwords, losses)
    dialogue = "".join(subwords).replace("Ġ", " ").replace("<|endoftext|>", " <|endoftext|> ")  # subwords --> dialogue
    word_level_losses = "".join(losses).replace("Ġ", " ")
    assert len(dialogue.split()) == len(word_level_losses.split())
    words = dialogue.split()
    losses = average_loss(word_level_losses)
    assert len(words) == len(losses)
    return words, losses


def process_one(data):
    subwords = data[0]
    losses = data[1]
    summary = data[2]
    assert len(subwords) == len(losses)

    words, losses = recover_word_level(subwords, losses)  # recover word-level losses

    return [words, losses, summary]


def process(datas, dataset, mode):
    res = []
    for data in datas:
        res.append(process_one(data))

    if not os.path.exists("./loss/{}/word/".format(dataset)):
        os.makedirs("./loss/{}/word/".format(dataset))

    with open("./loss/{}/word/{}_loss.json".format(dataset, mode), 'w') as file_obj:
        json.dump(res, file_obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument('-d', type=str, default="samsum", help='dataset')

    args = parser.parse_args()
    dataset = args.d
    assert dataset == "samsum" or dataset == "ami"

    train_datas, valid_datas, test_datas = get_datas("./loss/{}/bpe/train_loss.json".format(dataset),
                                                     "./loss/{}/bpe/valid_loss.json".format(dataset),
                                                     "./loss/{}/bpe/test_loss.json".format(dataset))
    process(train_datas, dataset, "train")
    process(valid_datas, dataset, "valid")
    process(test_datas, dataset, "test")
