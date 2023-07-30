import os,sys
import re
sys.path.append(os.getcwd())
from nltk import word_tokenize
from utils.constants import *
from utils.helpers import *

@memoize
def get_stopwords():
    """
        Read deactivated words and memorize them for direct access next time.
    """
    stop_words=[]
    stop_words_file_path=os.path.join(os.path.dirname(__file__),"../dataset/story_stopwords.txt")
    with open(stop_words_file_path,'r') as f:
        for line in f:
            stop_words.append(line.strip('\n'))
    return stop_words

def split_deeply(tokenized_list):
    """
        Insufficient space segmentation, further segmentation is more thorough.
    """
    new_tokenized_list=[]
    delimiters = ".", "/", "_","-"
    regexPattern = '|'.join(map(re.escape, delimiters))
    for word in tokenized_list:
        new_tokenized_list.extend(re.split(regexPattern,word.lower().strip()))
    return  new_tokenized_list

def tokenize_and_remove_stopwords(phrase):
    """
        Segmenting semantic phrases in chart fact and removing stop words.
    """
    stop_words=get_stopwords()
    tokenized_text=split_deeply(word_tokenize(phrase))
    return [word for word in tokenized_text if word not in stop_words] 


def extract_semantic_info(fact):
    """
        Read deactivated words and memorize them for direct access next time.
        Inputs:
            `fact`: chart fact, consisting of 7 tuples, is used to represent a chart
    """
    stop_words=get_stopwords()
    tokenized_semantic_token=list()
    semantic_pos_list=list()
    # -----------subspace, breakdown, measure, focus, meta-----------
    # subspace (field + value)
    subspace=fact["subspace"]
    for item in subspace:
        cut_tokenized_text_field = tokenize_and_remove_stopwords(item["field"])
        cut_tokenized_text_value = tokenize_and_remove_stopwords(item["value"])

        tokenized_semantic_token.extend(cut_tokenized_text_field)
        semantic_pos_list.extend([SEMANTIC_POS["subspace-field"]]*len(cut_tokenized_text_field))

        tokenized_semantic_token.extend(cut_tokenized_text_value)
        semantic_pos_list.extend([SEMANTIC_POS["subspace-value"]]*len(cut_tokenized_text_value))
  
    # breakdown (field)
    breakdown=fact["breakdown"]
    for item in breakdown:
        cut_tokenized_text_field = tokenize_and_remove_stopwords(item["field"])
        tokenized_semantic_token.extend(cut_tokenized_text_field)
        semantic_pos_list.extend([SEMANTIC_POS["breakdown-field"]]*len(cut_tokenized_text_field))
    
    # measure (field)
    measure=fact["measure"]
    for item in measure:
        cut_tokenized_text_field = tokenize_and_remove_stopwords(item["field"])
        tokenized_semantic_token.extend(cut_tokenized_text_field)
        semantic_pos_list.extend([SEMANTIC_POS["measure-field"]]*len(cut_tokenized_text_field))

    # focus (field + value)
    focus=fact["focus"]
    for item in focus:
        cut_tokenized_text_field = tokenize_and_remove_stopwords(item["field"])
        tokenized_semantic_token.extend(cut_tokenized_text_field)
        semantic_pos_list.extend([SEMANTIC_POS["focus-field"]]*len(cut_tokenized_text_field))
        
        if len(item["value"])<30:
            tokenized_text_value=split_deeply(word_tokenize(item["value"]))
            cut_tokenized_text_value = [word for word in tokenized_text_value if word not in stop_words] 
            tokenized_semantic_token.extend(cut_tokenized_text_value)
            semantic_pos_list.extend([SEMANTIC_POS["focus-value"]]*len(cut_tokenized_text_value))

    # meta
    f_meta=fact["meta"]
    if fact["fact_type"]=="rank":
        for m in f_meta:
            if not isinstance(m,int) and not isinstance(m,float):
                cut_tokenized_text_field = tokenize_and_remove_stopwords(m)
                tokenized_semantic_token.extend(cut_tokenized_text_field)
                semantic_pos_list.extend([SEMANTIC_POS["meta"]]*(len(cut_tokenized_text_field)))

    elif f_meta!="":
        cut_tokenized_text_field = tokenize_and_remove_stopwords(f_meta)
        tokenized_semantic_token.extend(cut_tokenized_text_field)
        semantic_pos_list.extend([SEMANTIC_POS["meta"]]*(len(cut_tokenized_text_field)))

    return tokenized_semantic_token,semantic_pos_list

if __name__ =="__main__":
    import json
    with open("./dataset/training_data.json","r") as f:
        facts=json.load(f)
    for quad in facts:     
        for key in quad:       
            extract_semantic_info(quad[key])

    