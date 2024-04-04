import torch
from simple_bigram import BigramLanguageModel
from transformer import TransformerModel
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
torch.manual_seed(1337)

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: [itos[i] for i in l]

#print(encode("hii there"))
#print(decode(encode("hii there")))

data = torch.tensor(encode(text), dtype=torch.long)
data = data.to(device)

# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


block_size = 32
batch_size = 16
eval_iters = 200
eval_interval = 100
max_iters = 5000

def get_batch(split):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    return x,y



m = TransformerModel(vocab_size,device).to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

optimizer = torch.optim.AdamW(m.parameters(),lr=1e-3)

for steps in range(max_iters):
    if steps % eval_interval == 0 or steps == max_iters - 1:
        losses = estimate_loss()
        print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xt,yt = get_batch('train')
    #losses = estimate_loss()
    #print(f"step {steps}: train loss {losses['train']:.4f} , val loss {losses['val']:.4f}")
    logits,loss = m(xt,yt)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(''.join(decode(m.generate(context, max_new_tokens=2000)[0].tolist())))