# -*- coding: utf-8 -*-
import codecs
import json

from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model(name):
    model = AutoModelForCausalLM.from_pretrained(name)
    return model


def get_tokenizer(name):
    # tokenizer = AutoTokenizer.from_pretrained(name,force_download=True,resume_download=True)
    tokenizer = AutoTokenizer.from_pretrained(name)
    return tokenizer


def load_json(file_path):
    with codecs.open(file_path, "r", "utf-8") as f:
        datas = json.load(f)
    print("Load {} finished, Data size:{}".format(file_path.split("/")[-1], len(datas)))
    return datas


def get_dialogue_summary_pairs(train_datas, valid_datas, test_datas):
    train_pairs = []
    for data in train_datas:
        summary = data[0].strip()
        dialogue = data[1]
        utterances = dialogue.split("\n")
        if utterances[-1] == ":":
            utterances = utterances[:-1]
        train_pairs.append([utterances, summary])

    valid_pairs = []
    for data in valid_datas:
        summary = data[0].strip()
        dialogue = data[1]
        utterances = dialogue.split("\n")
        if utterances[-1] == ":":
            utterances = utterances[:-1]
        valid_pairs.append([utterances, summary])

    test_pairs = []
    for data in test_datas:
        summary = data[0].strip()
        dialogue = data[1]
        utterances = dialogue.split("\n")
        if utterances[-1] == ":":
            utterances = utterances[:-1]
        test_pairs.append([utterances, summary])

    return train_pairs, valid_pairs, test_pairs

def get_datas(train_path, valid_path, test_path):
    train_datas = load_json(train_path)
    valid_datas = load_json(valid_path)
    test_datas = load_json(test_path)
    return train_datas, valid_datas, test_datas


def extract_top_k_index(losses, topk):
    indices = sorted(range(len(losses)), key=lambda i: losses[i])[-topk:]
    indices.sort()  # recover order
    return indices