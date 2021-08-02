# -*- coding: utf-8 -*-
import argparse
import codecs
import json
import os
from string import punctuation

from utils import get_datas, extract_top_k_index


def get_sim_datas(dataset, mode):
    f = codecs.open("./rep/{}/sim/{}_sim.json".format(dataset, mode), "r", encoding="utf-8")
    lines = f.readlines()
    datas = [json.loads(l) for l in lines]
    print("Load {} finished, Data size:{}".format(mode, len(datas)))
    return datas


def get_pure_dialogue_content(words, losses, speakers):
    clean_words = []
    clean_losses = []
    for word, loss in zip(words, losses):
        if word in speakers and (loss == 100 or loss == -100):
            continue
        if word == ":" and (loss == 100 or loss == -100):
            continue
        clean_words.append(word)
        clean_losses.append(loss)
    return clean_words, clean_losses


def get_speakers(words):
    """
    Get all speaker names from a dialogue
    """
    speakers = []
    utterances = " ".join(words).split("<|endoftext|>")[:-1]
    for u in utterances:
        s = u.split(":")[0].strip()  # select names according to ":"
        if s:
            speakers.append(s)
    return list(set(speakers))


def remove_speaker_and_first_u(words, losses, speakers):
    """
    Remove speaker names and the first utterance from dialogue
    """
    first_u = []
    clean_words = []
    clean_losses = []
    for word, loss in zip(words, losses):
        if word in speakers and (loss == 100 or loss == -100):
            continue
        if word == ":" and (loss == 100 or loss == -100):
            continue
        if loss == 100:
            first_u.append(word)
            continue

        clean_words.append(word)
        clean_losses.append(loss)
    return clean_words, clean_losses, first_u


def remove_first_word(words, losses):
    first_words = []
    clean_words = []
    clean_losses = []

    first_w_indices = []
    for index, word in enumerate(words):
        if word == "<|endoftext|>":
            first_w_indices.append(index + 1)

    for index, (word, loss) in enumerate(zip(words, losses)):
        if word == "<|endoftext|>":
            continue
        if index in first_w_indices:
            first_words.append(word)
            continue

        clean_words.append(word)
        clean_losses.append(loss)
    return clean_words, clean_losses, first_words


def get_keywords_num(words, ratio):
    return int(len(words) * ratio)


def get_topics_num(utterances, ratio):
    return int(len(utterances) * ratio)


def keywords_extraction(data, ratio):
    words = data[0]
    losses = data[1]

    keywords = []
    speakers = get_speakers(words)  # extract speaker names
    keywords.extend(speakers)

    # get pure dialogue content
    words, losses, first_u = remove_speaker_and_first_u(words, losses, speakers)
    keywords.extend(first_u)

    # remove the first word of each utterance
    words, losses, first_words = remove_first_word(words, losses)

    # dynamically get top-k words
    top_k = get_keywords_num(words, ratio)

    # extract top-k
    indices = extract_top_k_index(losses, top_k)
    selected_keywords = [words[index] for index in indices]

    # add the first utterance
    keywords.extend(selected_keywords)

    # filter
    res = [w for w in keywords if w not in punctuation]

    return res


def get_loss_for_each_utterance(words, losses):
    assert len(words) == len(losses)

    utterances = []
    utterances_loss = []

    utterance = []
    utterance_loss = []

    for word, loss in zip(words, losses):

        if word == "<|endoftext|>":
            if len(utterance_loss) == 0:
                utterance_loss = [0]
            utterances.append(utterance)
            utterances_loss.append(utterance_loss)
            utterance = []
            utterance_loss = []
        else:
            utterance.append(word)
            utterance_loss.append(loss)
    assert len(utterances) == len(utterances_loss)

    loss_for_each_u = [sum(utterance_loss) / len(utterance_loss) for utterance_loss in utterances_loss]
    return utterances, loss_for_each_u


def topic_segmentation(utterances, loss_for_each_utterance, topic_ratio):
    i_u_l = [(index, u, loss) for index, (u, loss) in enumerate(zip(utterances, loss_for_each_utterance))]

    topics_num = get_topics_num(utterances, topic_ratio)

    # get index-->loss map
    index2loss = dict()
    for index, loss in enumerate(loss_for_each_utterance):
        index2loss[index] = loss

    new_i_u_l = sorted(i_u_l, key=lambda x: x[-1], reverse=True)  # sort by loss in descending order

    seg_ids = []

    for triple in new_i_u_l:

        if len(seg_ids) == topics_num:
            break

        index = triple[0]

        if index == 0 or index == 1:  # do not consider 1st and 2nd
            continue
        else:
            seg_ids.append(index)

    return seg_ids


def get_topic_segment_indices(data, topic_ratio):
    words = data[0]
    losses = data[1]

    speakers = get_speakers(words)
    words, losses = get_pure_dialogue_content(words, losses, speakers)
    utterances, loss_for_each_utterance = get_loss_for_each_utterance(words, losses)

    seg_ids = topic_segmentation(utterances, loss_for_each_utterance, topic_ratio)

    return seg_ids


def rdd(data, threshold):
    """retain Chinese comments"""
    utterances = data[0]
    similarity_matrix = data[1]
    summary = data[2]
    assert len(similarity_matrix) + 1 == len(utterances)

    redundancy_indices = []

    true_indices = [i for i in range(len(utterances) - 1, 0, -1)]  # 相似度矩阵是先存储的最后一个句子和之前的相似度 因此倒序索引

    for index_i, sims in enumerate(similarity_matrix):  # 从最后一个句子和之前句子的相似度开始迭代
        q_index = true_indices[index_i]  # 获得当前句子的真实位置 处理的第0个 就是对话中的最后一个句子
        for index_j, sim in enumerate(reversed(sims)):  # sim 是当前句子对于其他句子的相似度计算，而且其他句子是正序排序的 例如第5句话和1234的相似度值 因此需要从后往前考虑
            if sim >= threshold:  # 如果相似度超过了阈值，我们认为再加上后面的句子之后，对于句子的表示并没有什么影响，因此新加上的句子并不重要
                if (q_index - index_j) not in redundancy_indices:
                    redundancy_indices.append(
                        q_index - index_j)  # 获得实际的冗余句子索引，q_index是当前查询句的索引，我们又是倒序一个一个向前处理句子，因此q_index-index_j代表了新加入的句子
            else:
                # 如果没有超过阈值，所以新加入的句子引入了新的信息。直接开始处理下一句（更考前的句子）
                break

    redundancy_indices = sorted(redundancy_indices)
    return utterances, redundancy_indices, summary


def create_dialogue(utterances, segment_ids, redundancy_ids):
    dialogue = []
    for index, utterance in enumerate(utterances):
        if index in segment_ids:
            dialogue.append("[TS]")
        if index in redundancy_ids:
            words = utterance.split()
            assert words[1] == ":"
            words.insert(2, "[RD]")
            utterance = " ".join(words)
        dialogue.append(utterance)

    return " <|endoftext|> ".join(dialogue)


def process(datas, sim_datas, ratio, threshold, topic_ratio, dataset, mode):
    res = []
    for data, sim_data in zip(datas, sim_datas):
        output = data[2]  # summary

        """Keywords Extraction"""
        keywords = keywords_extraction(data, ratio)

        """Topic Segmentation"""
        segment_ids = get_topic_segment_indices(data, topic_ratio)

        """Redundancy Detection"""
        utterances, redundancy_indices, _ = rdd(sim_data, threshold=threshold)

        """combine"""
        dialogue = create_dialogue(utterances, segment_ids, redundancy_indices)
        input = dialogue + " #KEY# " + " ".join(keywords)
        input = input.replace("<|endoftext|>", "|")

        res.append([input, output])

    if not os.path.exists("./data/{}/final".format(dataset)):
        os.makedirs("./data/{}/final".format(dataset))

    with open("./data/{}/final/{}.json".format(dataset, mode), 'w') as file_obj:
        json.dump(res, file_obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument('-d', type=str, default="samsum", help='dataset')

    args = parser.parse_args()
    dataset = args.d
    assert dataset == "samsum" or dataset == "ami"

    train_datas, valid_datas, test_datas = get_datas("./loss/{}/word/train_loss.json".format(dataset),
                                                     "./loss/{}/word/valid_loss.json".format(dataset),
                                                     "./loss/{}/word/test_loss.json".format(dataset))

    train_sim_datas = get_sim_datas(dataset, "train")
    valid_sim_datas = get_sim_datas(dataset, "valid")
    test_sim_datas = get_sim_datas(dataset, "test")

    if dataset == "samsum":
        process(train_datas, train_sim_datas, ratio=0.15, threshold=0.99, topic_ratio=0.15, dataset=dataset,
                mode="train")
        process(valid_datas, valid_sim_datas, ratio=0.15, threshold=0.99, topic_ratio=0.15, dataset=dataset,
                mode="valid")
        process(test_datas, test_sim_datas, ratio=0.15, threshold=0.99, topic_ratio=0.15, dataset=dataset, mode="test")
    elif dataset == "ami":
        process(train_datas, train_sim_datas, ratio=0.04, threshold=0.95, topic_ratio=0.05, dataset=dataset,
                mode="train")
        process(valid_datas, valid_sim_datas, ratio=0.04, threshold=0.95, topic_ratio=0.05, dataset=dataset,
                mode="valid")
        process(test_datas, test_sim_datas, ratio=0.04, threshold=0.95, topic_ratio=0.05, dataset=dataset, mode="test")
