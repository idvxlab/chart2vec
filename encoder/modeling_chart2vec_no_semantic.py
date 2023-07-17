"""
    Author: ChYing
    1. structural + semantic
    2. CNN + Bert
    3. concat firstly + 2FC
    4. dataset: 2positive + 1negative (triplets)
"""
import torch
import torch.nn as nn
from .utils.constants import *


class StructuralEmbedding(nn.Module):
    """
       According to the grammar tree, the structure of the fact is encoded, and CNN is used. 
    """
    def __init__(self, rules_feature=MAX_STRUCT_FEATURE_LEN,out_channels=8,kernel_size=3):
        super(StructuralEmbedding, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(rules_feature, out_channels, kernel_size, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.Flatten()
        )

    def forward(self, chart_struct):
        return self.conv(chart_struct)

class Chart2Vec(nn.Module):
    """
        Chart2vec model.
        Based on BERT, capturing chart information from both structural and semantic aspects.
        Inputs:
            `struct_one_hot`: numpy.array(), Grammar tree one-hot encoding.
            `indexed_tokens`: numpy.array(), The word index obtained by the bert tokenizer, which can be used to obtain the word vector later.
            `fact_pos_id`: numpy.array(), Identify the fact token location.
    """
    def __init__(self, rep_size=400, hidden_dropout_prob=0.1,last_rep_size=300):
        super(Chart2Vec, self).__init__()
        self.structual_embedding=StructuralEmbedding()
        
        self.fc=nn.Sequential(
            nn.Linear(rep_size,rep_size),
            nn.BatchNorm1d(rep_size),
            nn.ReLU(inplace=True),
            nn.Linear(rep_size,last_rep_size) 
        )
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.batchNorm=nn.BatchNorm1d(last_rep_size)
        # self.layernorm=nn.LayerNorm(last_rep_size)
        

    def forward(self, struct_one_hot,indexed_tokens,fact_pos_id):
        struct_one_hot=torch.FloatTensor(struct_one_hot).to(device)
        structual_embed=self.structual_embedding(struct_one_hot)
        # concat  + 2 dense
        fusion_tensor=self.fc(structual_embed)
        fusion_tensor=self.batchNorm(fusion_tensor)
        fusion_tensor=self.dropout(fusion_tensor)

        return fusion_tensor


