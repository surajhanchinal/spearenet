from tinygrad.tensor import Tensor
from tinygrad import nn
from typing import Tuple,List,Callable
import numpy as np
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tqdm import trange

n_heads = 4
n_embeds = 64
head_size = n_embeds // n_heads
dropout = 0.0
batch_size = 16
block_size = 32
n_layers = 4

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: [itos[i] for i in l]


data = Tensor(encode(text),requires_grad=False)
print(data.shape)
# Let's now split up the data into train and validation sets
n = int(0.9*data.shape[0]) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split) -> Tuple[Tensor,Tensor]:
    data = train_data if split == 'train' else val_data
    #idx = Tensor.randint(block_size,low=0,high=data.shape[0]-block_size)
    idx = np.random.randint(0,data.shape[0] - block_size,batch_size)
    x = Tensor.stack([data[i:i+block_size] for i in idx])
    y = Tensor.stack([data[i+1:i+block_size+1] for i in idx])
    return x,y


class Head:
    def __init__(self):
        self.key = nn.Linear(n_embeds,head_size,bias=False)
        self.query = nn.Linear(n_embeds,head_size,bias=False)
        self.value = nn.Linear(n_embeds,head_size,bias=False)
        self.tril = Tensor.ones(block_size,block_size).tril(block_size)
        self.tril.requires_grad = False

    def __call__(self,x:Tensor):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        weights = q.matmul(k.transpose(-2,-1)) * (C**-0.5)
        weights = weights.softmax()
        return weights.matmul(v).dropout(dropout)
    

class MultiHeadAttention:
    def __init__(self):
        self.heads = [Head() for _ in range(n_heads)]
        self.fc = nn.Linear(n_embeds,n_embeds)

    def __call__(self,x:Tensor):
        out = Tensor.stack([h(x) for h in self.heads],dim=-1)
        out = out.reshape(out.shape[0],block_size,n_embeds)
        return self.fc(out).dropout(dropout)


class FeedForward:
    def __init__(self):
        self.layers = [
            nn.Linear(n_embeds,4*n_embeds),
            Tensor.relu,
            nn.Linear(4*n_embeds,n_embeds),
        ]
    
    def __call__(self,x:Tensor) -> Tensor:
        return x.sequential(self.layers)


class Block:
    def __init__(self):
        self.ln1 = nn.LayerNorm(n_embeds)
        self.ln2 = nn.LayerNorm(n_embeds)
        self.sa = MultiHeadAttention()
        self.ffw = FeedForward()
    
    def __call__(self,x:Tensor):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x


class TransformerModel:
    def __init__(self,vocab_size):
        self.token_embedding_table = nn.Embedding(vocab_size,n_embeds)
        self.position_embedding_table = nn.Embedding(block_size,n_embeds)
        self.layers: List[Callable[[Tensor], Tensor]] = [Block() for _ in range(n_layers)]
        self.fc = nn.Linear(n_embeds,vocab_size)
        self.ln = nn.LayerNorm(n_embeds)
        self.indices = Tensor.arange(block_size).reshape(1,block_size)
        self.indices.requires_grad = False

    def __call__(self,x:Tensor) -> Tensor:
        tok_emb = self.token_embedding_table(x)
        pos_emb = self.position_embedding_table(self.indices)
        x = tok_emb+pos_emb
        x = x.sequential(self.layers)
        x = self.ln(x)
        logits = self.fc(x)
        return logits

    def generate(self, idx, max_new_tokens):
        toks = np.zeros((1,block_size))
        for i in range(max_new_tokens):
            print(i)
            x = Tensor(toks[:,-block_size:])
            logits = self(x)
            logits = logits[0, -1, :] # becomes (B, C)
            probs = Tensor.softmax(logits,axis=-1) # (B, C)
            idx_next = Tensor.multinomial(probs, num_samples=1) # (B, 1)
            toks = np.append(toks,[idx_next.item()]).reshape((1,-1))
        return toks
    


if __name__ == "__main__":
    model = TransformerModel(vocab_size)
    parameters = nn.state.get_parameters(model)
    
    opt = nn.optim.AdamW(parameters)
    tarange = Tensor.arange(vocab_size)
    tarange.requires_grad = False
    def get_loss(x:Tensor,y:Tensor) -> Tensor:
        logits = model(x)
        B,T,C = logits.shape
        logits = logits.reshape(B*T,C)
        probs = logits.log_softmax(axis=-1)
        y = y.reshape(B*T)
        
        mask = y.unsqueeze(1) == tarange
        loss = -(mask*probs).sum().div(B*T)
        return loss

    def train_step() -> Tensor:
        with Tensor.train():
            xt,yt = get_batch('train')
            loss = get_loss(xt,yt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            return loss
        
    def get_test_acc() -> Tensor:
        xv,yv = get_batch('val')
        return get_loss(xv,yv)

    
    test_acc = float('nan')
    for i in (t:=trange(100)):
        GlobalCounters.reset()   # NOTE: this makes it nice for DEBUG=2 timing
        loss = train_step()
        if i%10 == 9: test_acc = get_test_acc().item()
        t.set_description(f"training loss: {loss.item():.4f} validation loss: {test_acc:.4f} {GlobalCounters.mem_used}")


    #xc,yc = get_batch('val')
    #answer = model.generate(xc,10)
    #alist = [int(x) for x in answer[0].tolist()]
    #print(''.join(decode(alist)))