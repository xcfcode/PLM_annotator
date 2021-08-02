# -*- coding: utf-8 -*-
import argparse
import codecs
import json
import os

from tqdm import tqdm
from torch.nn import CrossEntropyLoss
import torch
import copy

from utils import get_model, get_tokenizer, get_dialogue_summary_pairs, load_json

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_datas(dataset):
    train_datas = load_json("./data/{}/train.json".format(dataset))
    valid_datas = load_json("./data/{}/valid.json".format(dataset))
    test_datas = load_json("./data/{}/test.json".format(dataset))
    return train_datas, valid_datas, test_datas


def combine_near_two_utterances(pairs):
    """
    Combine two adjacent utterances
    """
    dialogues = []
    for pair in pairs:  # for each dialogue-summary pair
        utterances = pair[0]  # list: [utterance_1, utterance_2, utterance_3,...,utterance_n]
        summary = pair[1]  # string
        utterance_num = len(utterances)
        # combine two adjacent utterances
        dialogue_combine_near_2 = [[utterances[i], utterances[i + 1]] for i in range(utterance_num - 1)]
        assert len(dialogue_combine_near_2) + 1 == len(utterances)
        dialogues.append([utterances, dialogue_combine_near_2, summary])
    return dialogues


def construct_example(context_subwords, response_subwords):
    """
    Construct teacher forcing input-output pair
    example abcd->efg
    we can get: abcd->e; abcde->f; abcdef->g
    """
    examples = []
    context_subwords.append("<|endoftext|>")
    for response_subword in response_subwords:
        input = copy.deepcopy(context_subwords)
        target = response_subword
        examples.append([input, target])
        context_subwords.append(response_subword)
    return examples


def process_one(dialogue, model, tokenizer):
    utterances = dialogue[0]
    dialogue_combine_2 = dialogue[1]
    summary = dialogue[2]

    """step-1 choose the first utterance"""
    first_utterance = utterances[0]
    first_utterance_subwords = tokenizer.tokenize(first_utterance)
    first_utterance_len = len(first_utterance_subwords)

    # init losses list
    processed_subwords = first_utterance_subwords + ["<|endoftext|>"]
    # the first utterance is always important, 100 means high loss.
    losses = [100 for _ in range(first_utterance_len)] + [-100]
    assert len(losses) == len(processed_subwords)

    """step-2 start to process utterance pairs"""
    for pair in dialogue_combine_2:
        context = pair[0]  # assume the first utterance is the context
        response = pair[1]  # assume the second utterance is the response

        context_subwords = tokenizer.tokenize(context)
        response_subwords = tokenizer.tokenize(response)

        # remove speaker
        new_context_subwords = context_subwords[context_subwords.index("Ġ:") + 1:]  # remove speaker
        response_speaker_index = response_subwords.index("Ġ:")
        new_response_subwords = response_subwords[response_speaker_index + 1:]  # remove speaker

        # construct teacher forcing examples
        examples = construct_example(new_context_subwords, new_response_subwords)

        # first append lossed for speaker: (since we remove speaker before)
        for i in range(response_speaker_index + 1):
            losses.append(-100)  # ignore speaker
        # extend this response
        processed_subwords.extend(response_subwords)

        for example in examples:
            input = example[0]
            target = example[1]

            input_ids = tokenizer.convert_tokens_to_ids(input)
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)  # [1, seq_len]
            target_ids = tokenizer.convert_tokens_to_ids(target)
            target_ids = torch.tensor(target_ids).unsqueeze(0).to(device)  # [1, 1]

            logits = model(input_ids, return_dict=True).logits[:, -1, :]  # [1, vocab_size]

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), target_ids)
            losses.append(loss.cpu().tolist())

        # post process
        processed_subwords.append("<|endoftext|>")
        losses.append(-100)

    assert len(processed_subwords) == len(losses)
    return processed_subwords, losses, summary


def process(dialogues, model, tokenizer, dataset, mode):
    res = []
    for dialogue in tqdm(dialogues):
        processed_subwords, losses, summary = process_one(dialogue, model, tokenizer)
        res.append([processed_subwords, losses, summary])

    if not os.path.exists("./loss/{}/bpe/".format(dataset)):
        os.makedirs("./loss/{}/bpe/".format(dataset))

    with open("./loss/{}/bpe/{}_loss.json".format(dataset, mode), 'w') as file_obj:
        json.dump(res, file_obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument('-d', type=str, default="samsum", help='dataset')

    args = parser.parse_args()
    dataset = args.d
    assert dataset == "samsum" or dataset == "ami"

    model = get_model("microsoft/DialoGPT-large")  # get model
    model.to(device)  # put to gpu
    tokenizer = get_tokenizer("microsoft/DialoGPT-large")  # get tokenizer

    train_datas, valid_datas, test_datas = get_datas(dataset)  # load data
    train_pairs, valid_pairs, test_pairs = get_dialogue_summary_pairs(train_datas, valid_datas, test_datas)

    train_dialogues = combine_near_two_utterances(train_pairs)
    valid_dialogues = combine_near_two_utterances(valid_pairs)
    test_dialogues = combine_near_two_utterances(test_pairs)

    process(test_dialogues, model, tokenizer, dataset, mode="test")
    process(valid_dialogues, model, tokenizer, dataset, mode="valid")
    process(train_dialogues, model, tokenizer, dataset, mode="train")
