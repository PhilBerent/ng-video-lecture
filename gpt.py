import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import time

torch.manual_seed(1337)

#wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# hyperparameters
runtype="live"
#runtype="test"
if runtype == "live":
    batch_size = 16 #3/16 how many independent sequences will we process in parallel?
    block_size = 32 #6/32 what is the maximum context length for predictions?
    n_embd = 64 #16/64
else:
    batch_size = 3 #3/16 how many independent sequences will we process in parallel?
    block_size = 6 #6/32 what is the maximum context length for predictions?
    n_embd = 16 #16/64

max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

n_head = 4
n_hembd = n_embd//n_head
n_layer = 4
dropout = 0.0
block_layer = 0
head_num = 0
tempOutputFile = r'C:\Users\Phil\Documents\Neural Nets\Code\ShakeGen\ShakeGen\tempoutput.txt'
# ------------


# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(block_size, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i-block_size+1:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class SlidingWindowMatrixMult3D(nn.Module):
    def __init__(self):
        super(SlidingWindowMatrixMult3D, self).__init__()
        self.keyWindowBias = nn.Parameter(torch.randn(1, 1, n_hembd, block_size))

    def forward(self, Q, K):
        # Get the dimensions of the input tensors
        B, T, C = Q.shape
        _, KT_dim, K_colm = K.shape

        # Check if Y has the expected shape
        assert KT_dim == 2 * T - 1, "Y shape is not as expected"

        # Create a sliding window view of Y_transposed with shape BxTxCxT
        K_sw = K.unfold(1, T, 1)
        K_sw = K_sw + self.keyWindowBias
        
        # Compute the Z matrix using einsum
        Z = torch.einsum('bwc,bwct->bwt', Q, K_sw)

        return Z

class SlidingWindowMatrixMult3D2(nn.Module):
    def __init__(self):
        super(SlidingWindowMatrixMult3D2, self).__init__()

    def forward(self, X, Y):
        # Get the dimensions of the input tensors
        B, T_X, T_Y = X.shape
        _, Ycol, C = Y.shape

        # Check if X and Y have the expected shapes
        assert T_X == T_Y, "X shape is not as expected"
        assert Ycol == 2 * T_Y - 1, "Y shape is not as expected"

        # Create a sliding window view of Y with shape BxTxTxC
        Y_sw = Y.unfold(1, T_Y, 1).permute(0, 1, 3, 2)

        # Compute the Z matrix using einsum
        Z = torch.einsum('btx,btxc->btc', X, Y_sw)

        return Z

class Head(nn.Module):
    """ one head of self-attention """
    # What this means is that each "head" processes input and calculates self dependencies differently
    # So 4 heads allow for the data to be looked at in four different ways

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.keyLayer1 = nn.Linear(n_embd, head_size, bias=False)
        self.keyICR = nn.Linear(n_embd, head_size, bias=False)
        self.keyCR = nn.Linear(n_embd, head_size, bias=False)
        self.vLayer1 = nn.Linear(n_embd, head_size, bias=False)
        self.vICR = nn.Linear(n_embd, head_size, bias=False)
        self.vCR = nn.Linear(n_embd, head_size, bias=False)
        #self.key = nn.Linear(block_size-1, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.sliding_window_matrix_mult_3D = SlidingWindowMatrixMult3D()
        self.sliding_window_matrix_mult_3D2 = SlidingWindowMatrixMult3D2()
        self.dropout = nn.Dropout(dropout)
        # self.posAdj = nn.Parameter(torch.randn(n_head, block_size, block_size))


    def forward(self, x):
        global head_num
        global block_layer
        B,T,C = x.shape
        xToPred = x[:, -block_size:, :]
        xPast = x[:,:block_size-1, :]
        if block_layer == 0:
            k = self.keyLayer1(x)
            v = self.vLayer1(x) 
        else:
            kCr = self.keyCR(xPast)
            kICR=self.keyICR(xToPred)
            k=torch.cat((kCr, kICR), 1)
            vCr = self.vCR(xPast)
            vICR=self.vICR(xToPred)
            v=torch.cat((vCr, vICR), 1)

        q = self.query(xToPred) # (B,T,C)
        weiNew = self.sliding_window_matrix_mult_3D(q, k)
        # weiNew = weiNew + self.posAdj[head_num]
        weiNew = F.softmax(weiNew, dim=-1) # (B, T, T)
        outNew = self.sliding_window_matrix_mult_3D2(weiNew, v)
        head_num += 1
        return outNew

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size
        self.num_heads = num_heads

    def forward(self, x):
        global head_num
        head_num = 0
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        outtemp = self.proj(out)
        out = self.dropout(outtemp)
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.n_head = n_head
        self.head_size = head_size
        self.sa = MultiHeadAttention(n_head, head_size) 
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)  # normalizes the layer using in sample mean and sd multiplying by gamma (trainable)
                                         # and adding beta (trainable)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        global block_layer
        xNorm = self.ln1(x)   # Normalizes. Input 'x' is 16x32x64. This normalizes each batch has 32 charachters each of which
                              # has 64 channels. This normalizes accross the last dim so that the mean and variance of the
                              # 64 numbers representing each letter is initially 0/1
        xpred = x[:,-block_size:,:]
        xPast = x[:,:block_size-1,:]
        xmh = self.sa(xNorm) # x passed through MultiHeadAttention()
        xmhPlusPred =  xpred + xmh
        xmhNorm = self.ln2(xmhPlusPred)
        xmhNormFed = self.ffwd(xmhNorm)
        xtemp4 = xmhPlusPred + xmhNormFed
        xout = torch.cat((xPast, xtemp4), dim=1)
        block_layer += 1
        return xout


# super simple bigram model
class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        #self.position_embedding_table = nn.Embedding(block_size*2-1, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        global block_layer
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        #pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb #+ pos_emb
        block_layer = 0
        #x = self.blocks(x)
        for i, block in enumerate(self.blocks):  # Iterate through blocks
            x = block(x)  # Pass the current iteration as an additional argument

        xpred = x[:,-block_size:,:]
        xpred = self.ln_f(xpred)
        logits = self.lm_head(xpred)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -(block_size*2-1):]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


def write_tmat_to_file(tensor):
    # Convert tensor to NumPy array
    tensor_np = tensor.detach().numpy()

    # Open file for writing
    with open(tempOutputFile, 'w') as f:
        print("File opened successfully")

        if tensor.dim() == 1:
            # Write 1D vector elements separated by tabs
            for elem in tensor_np:
                f.write(str(float(elem)) + '\t')
            # End with a newline character
            f.write('\n')

        elif tensor.dim() == 2:
            # Loop over rows in tensor
            for row in tensor_np:
                # Write each element in row to file separated by tabs
                for elem in row:
                    f.write(str(float(elem)) + '\t')  # Convert tensor element to float before writing
                # End row with a newline character
                f.write('\n')

        else:
            raise ValueError("The input tensor should have either 1 or 2 dimensions.")

    # Close file
    f.close()

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

#db
debug = 0
if debug==1:
    ix = torch.randint(block_size, len(data) - block_size, (1,))
    context = torch.stack([data[i-block_size+1:i+block_size] for i in ix])
    print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
#ed

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#max_iters=2
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
ix = torch.randint(block_size, len(data) - block_size, (1,))
context = torch.stack([data[i-block_size+1:i+block_size] for i in ix])
context = context.to(device)
#context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
