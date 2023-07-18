import numpy as np
import os,sys
sys.path.append(os.getcwd())
from utils.constants import *
from utils.helpers import *
import json


# 根据rules处理输入fact的格式
def fact_format(fact):
    new_fact = {}
    new_fact['fact_type'] = fact['fact_type']
    new_fact['chart_type'] = fact['chart_type']

    if fact['breakdown']:
        new_fact['breakdown'] = {}
        if len(fact['breakdown']) == 1:
            new_fact['breakdown']['len'] = "1"
        else:
            new_fact['breakdown']['len'] = '>1'
        new_fact['breakdown']['field'] = fact['breakdown'][0]['type']
    else:
        new_fact['breakdown'] = None

    # 目录是没有measure的
    if fact['measure']:
        new_fact['measure'] = {}
        if len(fact['measure']) == 1:
            new_fact['measure']['len'] = "1"
        else:
            new_fact['measure']['len'] = '>1'
        new_fact['measure']['field'] = fact['measure'][0]['type']
        new_fact['measure']['aggregate'] = fact['measure'][0]['aggregate']
        if 'subtype' in fact['measure'][0].keys():
            temp_subtype=fact['measure'][0]['subtype']
            if temp_subtype=="none":
                new_fact['measure']['subtype'] = None
            else:
                new_fact['measure']['subtype'] = temp_subtype
        else:
            new_fact['measure']['subtype'] = None
    else:
        new_fact['measure'] = None

    if fact['subspace']:
        new_fact['subspace'] = {}
        if len(fact['subspace']) == 1:
            new_fact['subspace']['len'] = "1"
            new_fact['subspace']['field'] = fact['subspace'][0]['type']
        else:
            new_fact['subspace']['len'] = '>1'
            new_fact['subspace']['field'] = 'multiple'

    else:
        new_fact['subspace'] = None

    # value 要重新提取到语义信息里面去
    if fact['focus']:
        new_fact['focus'] = {}
        if len(fact['focus']) == 1:
            new_fact['focus']['len'] = "1"
        else:
            new_fact['focus']['len'] = '>1'
        new_fact['focus']['field'] = fact['focus'][0]['type']
    else:
        new_fact['focus'] = None

    return new_fact

def get_rules(node, parentkey, rules):
    current_rule = parentkey + ' -> ' + ' "+" '.join(sorted(node.keys()))
    rules.append(current_rule)

    for k in sorted(node.keys()):
        v = node[k]
        if type(v) is dict:
            get_rules(v, k, rules)
        else:
            temp_rule=k + ' -> ' + '"' + str(v) + '"'
            if temp_rule not in rules:
                rules.append(temp_rule)

# 为fact生成onehot
@memoize
def get_total_rules():
    rule_path=os.path.join(os.path.dirname(__file__),'structural_rules.txt')
    rules = []
    with open(rule_path, 'r') as inputs:
        for line in inputs:
            line = line.strip()
            rules.append(line)
    rule2index = {}
    for i, r in enumerate(rules):
        rule2index[r] = i
    return rule_path,rule2index


def extract_structual_info(fact):
    total_rules,rule2index=get_total_rules()
    fact_rules= [] 
    fact=fact_format(fact)
    get_rules(fact, 'root', fact_rules)
    
    one_hot = np.zeros((MAX_STRUCT_FEATURE_LEN, len(total_rules)), dtype=np.float32)
    indices = [rule2index[r] for r in fact_rules]
    one_hot[np.arange(len(indices)), indices] = 1
    one_hot[np.arange(len(indices), MAX_STRUCT_FEATURE_LEN), -1] = 1

    return one_hot

# if __name__=="__main__":
#     with open("./dataset/training_data.json","r") as f:
#         facts=json.load(f)
#     # 设置打印选项，将所有结果打印出来
#     np.set_printoptions(threshold=np.inf)
#     max_struct=-1
#     chart_types=[]
#     for sample in facts:
#         for key in sample:
#             # try:
#                 rules_len,fact_rules=extract_structual_info(sample[key])
#                 chart=sample[key]["chart_type"]
#                 if chart not in chart_types:
#                     chart_types.append(chart)
#                 if rules_len>max_struct:
#                     max_struct=rules_len
#                     print(fact_rules)
#             # except:
#                 # print(sample[key]["fact_id"])
#     print("最长规则数：",max_struct)
#     print(chart_types,len(chart_types))