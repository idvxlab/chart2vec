import numpy as np
import os,sys
sys.path.append(os.getcwd())
from nltk import word_tokenize
import re
from utils.constants import *
from utils.helpers import *

@memoize
def get_stopwords():
    stop_words=[]
    path=os.path.dirname(__file__)
    stop_words_file_path=os.path.join(path,"../dataset/story_stopwords.txt")
    with open(stop_words_file_path,'r') as f:
        for line in f:
            stop_words.append(line.strip('\n'))
    return stop_words


def extract_semantic_info(fact):
    stop_words=get_stopwords()
    semantic_list=list()
    tokenized_semantic_token=list()
    semantic_pos_list=list()
    # subspace, breakdown, measure, focus, meta(rank & categorization)
    # subspace (field + value)
    subspace=fact["subspace"]
    for item in subspace:
        semantic_list.append(item["field"])
        semantic_list.append(item["value"])
        
        tokenized_text_field=split_deeply(word_tokenize(item["field"]))
        tokenized_text_value=split_deeply(word_tokenize(item["value"]))
        cut_tokenized_text_field = [word for word in tokenized_text_field if word not in stop_words] 
        cut_tokenized_text_value = [word for word in tokenized_text_value if word not in stop_words] 

        tokenized_semantic_token.extend(cut_tokenized_text_field)
        semantic_pos_list.extend([SEMANTIC_POS["subspace-field"]]*len(cut_tokenized_text_field))

        tokenized_semantic_token.extend(cut_tokenized_text_value)
        semantic_pos_list.extend([SEMANTIC_POS["subspace-value"]]*len(cut_tokenized_text_value))

    
    # breakdown (field)
    breakdown=fact["breakdown"]
    for item in breakdown:
        semantic_list.append(item["field"])

        tokenized_text_field=split_deeply(word_tokenize(item["field"]))
        cut_tokenized_text_field = [word for word in tokenized_text_field if word not in stop_words] 
        
        tokenized_semantic_token.extend(cut_tokenized_text_field)
        semantic_pos_list.extend([SEMANTIC_POS["breakdown-field"]]*len(cut_tokenized_text_field))
    
    # measure (field)
    measure=fact["measure"]
    for item in measure:
        semantic_list.append(item["field"])
        tokenized_text_field=split_deeply(word_tokenize(item["field"]))
        cut_tokenized_text_field = [word for word in tokenized_text_field if word not in stop_words] 

        tokenized_semantic_token.extend(cut_tokenized_text_field)
        semantic_pos_list.extend([SEMANTIC_POS["measure-field"]]*len(cut_tokenized_text_field))

    # focus (field + value)
    focus=fact["focus"]
    for item in focus:
        semantic_list.append(item["field"])
        tokenized_text_field=split_deeply(word_tokenize(item["field"]))
        cut_tokenized_text_field = [word for word in tokenized_text_field if word not in stop_words] 
        tokenized_semantic_token.extend(cut_tokenized_text_field)
        semantic_pos_list.extend([SEMANTIC_POS["focus-field"]]*len(cut_tokenized_text_field))
        
        if len(item["value"])<30:
            semantic_list.append(item["value"])
            tokenized_text_value=split_deeply(word_tokenize(item["value"]))
            cut_tokenized_text_value = [word for word in tokenized_text_value if word not in stop_words] 
            tokenized_semantic_token.extend(cut_tokenized_text_value)
            semantic_pos_list.extend([SEMANTIC_POS["focus-value"]]*len(cut_tokenized_text_value))

    # meta (type==rank or categrization)
    f_type=fact["fact_type"]
    f_meta=fact["meta"]
    if f_type=="rank":
        for m in f_meta:
            if not isinstance(m,int) and not isinstance(m,float):
                semantic_list.append(m)
                tokenized_text_field=split_deeply(word_tokenize(m))
                cut_tokenized_text_field = [word for word in tokenized_text_field if word not in stop_words] 

                tokenized_semantic_token.extend(cut_tokenized_text_field)
                semantic_pos_list.extend([SEMANTIC_POS["meta"]]*(len(cut_tokenized_text_field)))

    elif f_meta!="":
        semantic_list.append(f_meta)
        tokenized_text_field=split_deeply(word_tokenize(f_meta))
        cut_tokenized_text_field = [word for word in tokenized_text_field if word not in stop_words] 

        tokenized_semantic_token.extend(cut_tokenized_text_field)
        semantic_pos_list.extend([SEMANTIC_POS["meta"]]*(len(cut_tokenized_text_field)))

    return tokenized_semantic_token,semantic_pos_list

def split_deeply(tokenized_list):
    """
        空格分割不充分，再次按照_和/进行分割
    """
    new_tokenized_list=[]
    delimiters = ".", "/", "_","-"
    regexPattern = '|'.join(map(re.escape, delimiters))
    for word in tokenized_list:
        new_tokenized_list.extend(re.split(regexPattern,word.lower().strip()))
   
    return  new_tokenized_list

# if __name__=="__main__":
#     import json
#     # with open("./dataset/training_data.json","r") as f:
#     #     facts=json.load(f)
#     # max_semantic_num=-1
#     # for sample in facts:
#     #     for key in sample:
#     #         fact=sample[key]
#     #         tokenized_semantic_token,semantic_pos_list=extract_semantic_info(sample[key])
#     #         temp_len=len(semantic_pos_list)
#     #         if fact["fact_type"]=="rank":
#     #             if not isinstance(fact["meta"],list):
#     #                 if fact["meta"]=="":
#     #                     fact["fact_type"]="distribution"
#     #                 else:
#     #                     fact["meta"]=fact["meta"].split(",")
#     #                 print(fact["fact_id"])

#     #         if temp_len>max_semantic_num:
#     #             max_semantic_num=temp_len
#     # write_json_list("./dataset/cleaned_training_data.json",facts)
#     # print(extract_semantic_info(facts[0]))
#     # print("最长语义数",max_semantic_num)

#     with open("./dataset/sub_training_data.json","r",encoding="utf-8") as f:
#         facts=json.load(f)
#     max_semantic_num=-1
#     for fact in facts:
#         tokenized_semantic_token,semantic_pos_list=extract_semantic_info(fact)
#         if fact["fact_type"]=="rank":
#             if not isinstance(fact["meta"],list):
#                 if fact["meta"]=="":
#                     fact["fact_type"]="distribution"
#                 else:
#                     fact["meta"]=fact["meta"].split(",")
#                 print(fact["fact_id"])

#         temp_len=len(semantic_pos_list)
#         if temp_len>max_semantic_num:
#             max_semantic_num=temp_len
#             print(fact["fact_id"],tokenized_semantic_token)
#     print("最长语义数",max_semantic_num)