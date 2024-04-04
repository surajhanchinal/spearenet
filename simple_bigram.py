import torch
import torch.nn as nn
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):

    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self,idx,targets=None):
        # idx and targets BATCH X BLOCK_SIZE (T)
        logits = self.token_embedding_table(idx) # B X T X C (C is vocab_size)

        if targets is None:
            loss = None
        else: 
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits,loss
    
    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            logits,loss = self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1) # B x T+1
        return idx
