import torch
MAX_SEMANTIC_LEN=25
MAX_STRUCT_FEATURE_LEN=16
SEMANTIC_POS={
    "NONE":0,
    "subspace-field":1,
    "subspace-value":2,
    "breakdown-field":3,
    "measure-field":4,
    "focus-field":5,
    "focus-value":6,
    "meta":7
}
WORD_VECTORS_LEN=100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')