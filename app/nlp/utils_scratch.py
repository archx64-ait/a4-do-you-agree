import json, os
import torch
from torch import nn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "webapp", "custom_bert_mnli.pth")
CHEAD_PATH = os.path.join(BASE_DIR, "webapp", "classifier_head.pth")
TOKEN_PATH = os.path.join(BASE_DIR, "webapp", "my_tokenizer.json")

with open(TOKEN_PATH, 'r') as f:
    tokenizer = json.load(f)

word2id = tokenizer['word2id']
id2word = tokenizer['id2word']
vocab_size = len(word2id)

max_len = 1000
n_layers = 12
n_heads = 12
d_model = 768
d_ff = d_model * 4
d_k = d_v = 64
n_segments = 2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 69
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def mean_pool(token_embeds, attention_mask):
    # reshape attention_mask to cover 768-dimension embeddings
    in_mask = attention_mask.unsqueeze(-1).expand(
        token_embeds.size()
    ).float()
    # perform mean-pooling but exclude padding tokens (specified by in_mask)
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-9
    )
    return pool

def custom_tokenizer(
        sentence, max_length=128, padding="max_length", truncation=True
    ):
        tokens = sentence.lower().split()  # basic tokenization by splitting words
        token_ids = [word2id.get(token, word2id["[UNK]"]) for token in tokens]
        attention_mask = [1] * len(token_ids)

        # Truncate if necessary
        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            attention_mask = attention_mask[:max_length]

        # Padding to max_length
        if padding == "max_length":
            padding_length = max_length - len(token_ids)
            token_ids += [word2id["[PAD]"]] * padding_length
            attention_mask += [0] * padding_length

        return {"input_ids": [token_ids], "attention_mask": [attention_mask]}


class Embedding(nn.Module):
    def __init__(self, vocab_size, max_len, n_segments, d_model, device):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(max_len, d_model)  # positional embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment embedding
        self.norm = nn.LayerNorm(d_model)  # layer normalization
        self.device = device

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = (
            torch.arange(seq_len, dtype=torch.long)
            .to(self.device)
            .unsqueeze(0)
            .expand_as(x)
        )  # create position indices
        embedding = (
            self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        )  # sum all embeddings
        return self.norm(embedding)  # apply layer normalization


class BERT(nn.Module):
    def __init__(
        self,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        d_k,
        n_segments,
        vocab_size,
        max_len,
        device,
    ):
        super(BERT, self).__init__()
        self.embedding = Embedding(
            vocab_size, max_len, n_segments, d_model, device
        )  # embedding layer
        self.layers = nn.ModuleList(
            [EncoderLayer(n_heads, d_model, d_ff, d_k, device) for _ in range(n_layers)]
        )  # transformer encoder layers
        self.fc = nn.Linear(d_model, d_model)  # fully connected layer for hidden states
        self.activ = nn.Tanh()  # activation function
        self.linear = nn.Linear(d_model, d_model)  # another linear layer
        self.norm = nn.LayerNorm(d_model)  # layer normalization
        self.classifier = nn.Linear(d_model, 2)  # classifier head for predictions
        self.decoder = nn.Linear(
            d_model, vocab_size, bias=False
        )  # decoder for language modeling
        self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))  # bias for decoder
        self.device = device

    def forward(self, input_ids, segment_ids):
        output = self.embedding(input_ids, segment_ids)  # get embeddings
        enc_self_attn_mask = get_attn_pad_mask(
            input_ids, input_ids, self.device
        )  # attention mask
        for layer in self.layers:
            output, _ = layer(
                output, enc_self_attn_mask
            )  # pass through transformer layers
        return output  # return hidden states

    def get_last_hidden_state(self, input_ids):
        segment_ids = torch.zeros_like(input_ids).to(
            self.device
        )  # default segment ids as zeros
        output = self.embedding(input_ids, segment_ids)  # get embeddings
        enc_self_attn_mask = get_attn_pad_mask(
            input_ids, input_ids, self.device
        )  # attention mask
        for layer in self.layers:
            output, _ = layer(
                output, enc_self_attn_mask
            )  # pass through transformer layers
        return output  # return last hidden state


def get_attn_pad_mask(seq_q, seq_k, device):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1).to(device)  # mask padding tokens
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # expand mask to all heads


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, d_k, device):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(
            n_heads, d_model, d_k, device
        )  # multi-head self-attention
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model, d_ff
        )  # position-wise feed-forward network

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask
        )  # self-attention mechanism
        enc_outputs = self.pos_ffn(enc_outputs)  # position-wise feed-forward
        return enc_outputs, attn  # return outputs and attention weights


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, device):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.W_Q = nn.Linear(d_model, d_k * n_heads)  # query projection
        self.W_K = nn.Linear(d_model, d_k * n_heads)  # key projection
        self.W_V = nn.Linear(d_model, d_k * n_heads)  # value projection
        self.device = device

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)  # residual connection
        q_s = (
            self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        )  # project queries
        k_s = (
            self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        )  # project keys
        v_s = (
            self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        )  # project values
        attn_mask = attn_mask.unsqueeze(1).repeat(
            1, self.n_heads, 1, 1
        )  # repeat mask for all heads
        context, attn = ScaledDotProductAttention(self.d_k, self.device)(
            q_s, k_s, v_s, attn_mask
        )  # scaled dot-product attention
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.d_k)
        )  # concatenate attention heads
        output = nn.Linear(self.n_heads * self.d_k, self.d_model).to(self.device)(
            context
        )  # final linear layer
        return (
            nn.LayerNorm(self.d_model).to(self.device)(output + residual),
            attn,
        )  # add & normalize


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, device):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = torch.sqrt(torch.FloatTensor([d_k])).to(device)  # scaling factor

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale  # scaled dot-product
        scores.masked_fill_(attn_mask, -1e9)  # apply attention mask
        attn = nn.Softmax(dim=-1)(scores)  # softmax to get attention weights
        context = torch.matmul(attn, V)  # weighted sum of values
        return context, attn  # return context and attention weights


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # first feed-forward layer
        self.fc2 = nn.Linear(d_ff, d_model)  # second feed-forward layer

    def forward(self, x):
        return self.fc2(nn.functional.gelu(self.fc1(x)))  # apply gelu activation
    


