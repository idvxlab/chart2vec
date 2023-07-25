"""
    Author: ChYing
    1. structural + semantic
    2. CNN + Bert
    3. concat firstly + 2FC
    4. dataset: 2positive + 1negative (triplets)
"""
import torch
import torch.nn as nn
import gensim.downloader as api
import numpy as np
import gensim
from utils.constants import *
from wikipedia2vec import Wikipedia2Vec


class Word2vecVector(nn.Module):
    """
        Based on the fact token, use BERT to get the word vector
        Inputs:
            `indexed_tokens`: The word index obtained by the bert tokenizer, which can be used to obtain the word vector later
    """
    def __init__(self,window_size=10):
        super(Word2vecVector, self).__init__()
        MODLE_FILE="/home/chenying2929/Chart2Vec-test-Wikipedia2vec/enwiki_20180420_win10_100d.pkl"
        self.word2vec_model = Wikipedia2Vec.load(MODLE_FILE)
        self.w_avg_pool=nn.AvgPool1d(window_size, stride=window_size)

    def forward(self, tokenized_text_list):
        word2vec_list=[]
        for sentence in tokenized_text_list:
            temp_sentence_list=[]
            for word in sentence:
                try:
                    word_vectors=self.word2vec_model.get_word_vector(word.lower())
                except:
                    word_vectors=[0]*WORD_VECTORS_LEN
                temp_sentence_list.extend(np.array(word_vectors))
            word2vec_list.append(temp_sentence_list)
        temp_list=np.array(word2vec_list)
        temp_sentence_tensor=torch.FloatTensor(temp_list).to(device)
        output=self.w_avg_pool(temp_sentence_tensor)
        return output


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
    def __init__(self, rep_size=10*MAX_SEMANTIC_LEN+400, hidden_dropout_prob=0.1):
        super(Chart2Vec, self).__init__()
        self.structual_embedding=StructuralEmbedding()
        self.semantic_embedding=SemanticEmbedding()

        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.batchNorm=nn.BatchNorm1d(rep_size)
        # self.layernorm=nn.LayerNorm(last_rep_size)
        

    def forward(self, struct_one_hot,indexed_tokens,fact_pos_id):
        struct_one_hot=torch.FloatTensor(struct_one_hot).to(device)
        structual_embed=self.structual_embedding(struct_one_hot)
        semantic_embed=self.semantic_embedding(indexed_tokens,fact_pos_id)
        # concat  + 2 dense
        concate_structure_semantic_tensor=torch.cat((structual_embed,semantic_embed),dim=1)
       
        fusion_tensor=self.batchNorm(concate_structure_semantic_tensor)
        fusion_tensor=self.dropout(fusion_tensor)

        return fusion_tensor


class SemanticEmbedding(nn.Module):
    """
        Obtained by adding up the semantic information of the token and the location of the semantics, and finally performing the pooling operation.
        Inputs:
            `indexed_token`: numpy.array(), The word index obtained by the bert tokenizer, which can be used to obtain the word vector later.
            `fact_pos_id`: numpy.array(), Identify the fact token location.
    """
    def __init__(self, fact_pos_num=8, hidden_size=10):
        super(SemanticEmbedding, self).__init__()
        self.w2v_embeddings = Word2vecVector()
        self.fact_pos_embeddings = nn.Embedding(fact_pos_num, hidden_size)

    def forward(self, indexed_token, fact_pos_id):
        fact_pos_id=torch.LongTensor(fact_pos_id).to(device)
        pos_emb = self.fact_pos_embeddings(fact_pos_id).to(device)
        pos_emb = torch.reshape(pos_emb, (pos_emb.size(0), pos_emb.size(1)*pos_emb.size(2)))
        token_emb = self.w2v_embeddings(indexed_token)
        # The corresponding positions are added together and averaged.
        semantic_emb=pos_emb+token_emb
        return semantic_emb
