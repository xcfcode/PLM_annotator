# -*- coding: utf-8 -*-


import codecs
import json
import os
from utils import get_model, get_tokenizer, get_dialogue_summary_pairs, load_json
from tqdm import tqdm
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_datas():
    train_datas = load_json("./data/ami/train.json")
    valid_datas = load_json("./data/ami/valid.json")
    test_datas = load_json("./data/ami/test.json")
    return train_datas, valid_datas, test_datas


def process_one(utterances, model, tokenizer):
    dialogue_subwords = []

    # remove speaker
    for utterance in utterances:
        utterance_subwords = tokenizer.tokenize(utterance)
        new_utterance_subwords = utterance_subwords[utterance_subwords.index("Ġ:") + 1:]
        dialogue_subwords.extend(new_utterance_subwords + ["<|endoftext|>"])

    indices_of_eos = [index for index, subword in enumerate(dialogue_subwords) if subword == "<|endoftext|>"]

    input_ids = tokenizer.convert_tokens_to_ids(dialogue_subwords)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)  # [1, seq_len]

    # choose the last hidden layer
    hidden_state = model(input_ids, return_dict=True, output_hidden_states=True).hidden_states[
        -1]  # [1, token_num, 1280]
    hidden_state = hidden_state.squeeze(0)  # [token_num, 1280]
    # choose the utterance representation
    utt_hidden_state = hidden_state[torch.tensor(indices_of_eos).to(device)]  # [utt_num, 1280]
    utt_reps = utt_hidden_state.cpu().tolist()
    return utt_reps


def split_utterances(utterances, inerval=5):
    return [utterances[i:i + inerval] for i in range(0, len(utterances), inerval)]


def process(pairs, model, tokenizer, mode):
    if not os.path.exists("./rep/ami"):
        os.makedirs("./rep/ami")

    file_obj = codecs.open("./rep/ami/{}_rep.json".format(mode), 'a', encoding="utf-8")

    for pair in tqdm(pairs):
        utterances = pair[0]
        summary = pair[1]

        segs = split_utterances(utterances, inerval=10)
        utt_reps = []
        for seg in segs:
            seg_rep = process_one(seg, model, tokenizer)
            utt_reps.extend(seg_rep)
        assert len(utt_reps) == len(utterances)
        json.dump([utterances, utt_reps, summary], file_obj)
        file_obj.write("\n")


if __name__ == "__main__":
    model = get_model("microsoft/DialoGPT-large")  # get model
    model.to(device)  # put to gpu

    tokenizer = get_tokenizer("microsoft/DialoGPT-large")  # get tokenizer

    train_datas, valid_datas, test_datas = get_datas()  # load data
    train_pairs, valid_pairs, test_pairs = get_dialogue_summary_pairs(train_datas, valid_datas, test_datas)

    process(test_pairs, model, tokenizer, mode="test")
    process(valid_pairs, model, tokenizer, mode="valid")
    process(train_pairs, model, tokenizer, mode="train")
