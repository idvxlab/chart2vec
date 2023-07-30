import copy
import sys
import os
import numpy as np
sys.path.append(os.getcwd())
from input_embedding.structual_extract import extract_structual_info
from input_embedding.semantic_extract import extract_semantic_info
from utils.constants import *


def quadruplet_batch_generator_word2vec(all_data, batch_size, shuffle=True):
    """
    Param:
        `all_data` : The entire dataset.
        `batch_size`: The size of each batch.
        `shuffle`: Whether to disrupt the order.
    Return:
        quadruplet_batch_data: anchor batch, positive1 batch, positive2 batch, negative batch
    """
    # Get the sample size.
    data_size = len(all_data)
    shuffle_data = []
    if shuffle:
        # Random generation of disrupted indexes.
        p = np.random.permutation(data_size)
        # Reorganize data.
        for i in p:
            shuffle_data.append(all_data[i])
        all_data = shuffle_data
    batch_count = 0
    while True:
        # One cycle is completed, disrupting the order once.
        if batch_count * batch_size + batch_size > data_size:
            batch_count = 0
            shuffle_data = []
            if shuffle:
                p = np.random.permutation(data_size)
                for i in p:
                    shuffle_data.append(all_data[i])
                all_data = shuffle_data
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        data_list = all_data[start:end]
        anchor_list, pos1_list, pos2_list, neg_list = [], [], [], []

        anchor_list, pos1_list, pos2_list, neg_list = zip(
            *[(value["chart1"], value["chart2"], value["chart3"], value["chart4"]) for value in data_list])
        input_lists = [anchor_list, pos1_list, pos2_list, neg_list]
        input_results = [w2v_convert_fact_to_input(lst) for lst in input_lists]

        yield padding_batch_quadruplet_data_w2v(
            input_results[0], input_results[1], input_results[2], input_results[3])

def w2v_convert_fact_to_input(chart_fact_list):
    """
        Transform the fact type batch input for model input.
    """
    batch_indexed_tokens = list()
    batch_pos = list()
    batch_struct_one_hot = list()
    max_token_len = 0
    for item_fact in chart_fact_list:
        tokenized_semantic_token, semantic_pos_list = extract_semantic_info(
            item_fact)
        strcuctual_one_hot = extract_structual_info(item_fact)
        if len(tokenized_semantic_token) > max_token_len:
            max_token_len = len(tokenized_semantic_token)
        batch_indexed_tokens.append(tokenized_semantic_token)
        batch_pos.append(semantic_pos_list)
        batch_struct_one_hot.append(strcuctual_one_hot)
    return {
        "batch_indexed_tokens": batch_indexed_tokens,
        "batch_pos": batch_pos,
        "batch_struct_one_hot": batch_struct_one_hot,
        "max_token_len": max_token_len
    }


def pad_w2v_convert_fact_to_input(chart_fact_list):
    """
        Transform the fact type batch input for model input.
    """
    batch_indexed_tokens = list()
    batch_pos = list()
    batch_struct_one_hot = list()
    max_token_len = 0
    for item_fact in chart_fact_list:
        tokenized_semantic_token, semantic_pos_list = extract_semantic_info(
            item_fact)
        strcuctual_one_hot = extract_structual_info(item_fact)
        if len(tokenized_semantic_token) > max_token_len:
            max_token_len = len(tokenized_semantic_token)
        batch_indexed_tokens.append(tokenized_semantic_token)
        batch_pos.append(semantic_pos_list)
        batch_struct_one_hot.append(strcuctual_one_hot)
    input = {
        "batch_indexed_tokens": batch_indexed_tokens,
        "batch_pos": batch_pos,
        "batch_struct_one_hot": batch_struct_one_hot,
    }
    return padding_one_batch_data_w2v(25, input)


def padding_batch_quadruplet_data_w2v(anchor_input, pos1_input, pos2_input, neg_input):
    max_token_num = max(anchor_input["max_token_len"],
                        pos1_input["max_token_len"], pos2_input["max_token_len"], neg_input["max_token_len"])
    max_token_num = MAX_SEMANTIC_LEN
    anchor_input = padding_one_batch_data_w2v(max_token_num, anchor_input)
    pos1_input = padding_one_batch_data_w2v(max_token_num, pos1_input)
    pos2_input = padding_one_batch_data_w2v(max_token_num, pos2_input)
    neg_input = padding_one_batch_data_w2v(max_token_num, neg_input)
    return anchor_input, pos1_input, pos2_input, neg_input


def padding_one_batch_data_w2v(max_token_num, chart_input):
    batch_indexed_tokens = chart_input["batch_indexed_tokens"]
    new_batch_indexed_tokens = list()
    for item in batch_indexed_tokens:
        temp_index_token = copy.deepcopy(item)
        if len(item) < max_token_num:
            sub = max_token_num-len(item)
            for i in range(0, sub):
                temp_index_token.append("")
        else:
            temp_index_token = temp_index_token[0:max_token_num]
        new_batch_indexed_tokens.append(temp_index_token)

    batch_pos = chart_input["batch_pos"]
    new_batch_pos = list()
    for item in batch_pos:
        temp_pos = copy.deepcopy(item)
        if len(item) < max_token_num:
            sub = max_token_num-len(item)
            for i in range(0, sub):
                temp_pos.append(0)
        else:
            temp_pos = temp_pos[0:max_token_num]
        new_batch_pos.append(temp_pos)
    return {
        "batch_indexed_tokens": new_batch_indexed_tokens,
        "batch_pos": np.array(new_batch_pos),
        "batch_struct_one_hot": np.array(chart_input["batch_struct_one_hot"])
    }