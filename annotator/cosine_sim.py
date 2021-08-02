# -*- coding: utf-8 -*-
import argparse
import codecs
import json
import os

from tqdm import tqdm
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_datas(dataset, mode):
    f = codecs.open("./rep/{}/{}_rep.json".format(dataset, mode), "r", encoding="utf-8")
    lines = f.readlines()
    datas = [json.loads(l) for l in lines]
    print("Load {} finished, Data size:{}".format(mode, len(datas)))
    return datas


def cosine_sim(q_rep, k_rep):
    q = torch.tensor(q_rep).unsqueeze(0).to(device)  # [1, 1280]
    k = torch.tensor(k_rep).to(device)  # [n, 1280]
    cos = torch.nn.CosineSimilarity(dim=1)
    weights = cos(q, k)
    return weights.cpu().tolist()


def process_one(data):
    utterances = data[0]
    reps = data[1]
    summary = data[2]

    weight_matrix = []

    for i in range(len(reps) - 1, 0, -1):
        q_rep = reps[i]
        k_rep = reps[:i]
        weights = cosine_sim(q_rep, k_rep)
        weight_matrix.append(weights)
    return utterances, weight_matrix, summary


def process(datas, dataset, mode):
    if not os.path.exists("./rep/{}/sim".format(dataset)):
        os.makedirs("./rep/{}/sim".format(dataset))

    file_obj = codecs.open("./rep/{}/sim/{}_sim.json".format(dataset, mode), 'a', encoding="utf-8")

    for data in tqdm(datas):
        utterances, weight_matrix, summary = process_one(data)
        json.dump([utterances, weight_matrix, summary], file_obj)
        file_obj.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument('-d', type=str, default="samsum", help='dataset')

    args = parser.parse_args()
    dataset = args.d
    assert dataset == "samsum" or dataset == "ami"

    train_datas = get_datas(dataset, "train")
    valid_datas = get_datas(dataset, "valid")
    test_datas = get_datas(dataset, "test")

    process(test_datas, dataset, "test")
    process(valid_datas, dataset, "valid")
    process(train_datas, dataset, "train")
