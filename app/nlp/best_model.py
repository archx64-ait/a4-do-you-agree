from torch import nn
import torch


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = AdditiveAttentionLayer(hid_dim, n_heads, dropout, device)
        self.feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len]   #if the token is padding, it will be 1, otherwise 0
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        # src: [batch_size, src len, hid dim]

        _src = self.feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))
        # src: [batch_size, src len, hid dim]

        return src


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hid_dim,
        n_layers,
        n_heads,
        pf_dim,
        dropout,
        device,
        max_length=500,
    ):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(self.device)

    def forward(self, src, src_mask):

        # src = [batch size, src len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = (
            torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        )
        # pos: [batch_size, src_len]

        src = self.dropout(
            (self.tok_embedding(src) * self.scale) + self.pos_embedding(pos)
        )
        # src: [batch_size, src_len, hid_dim]

        for layer in self.layers:
            src = layer(src, src_mask)
        # src: [batch_size, src_len, hid_dim]

        return src


class AdditiveAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout, device):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query_fc = nn.Linear(hidden_dim, hidden_dim)
        self.key_fc = nn.Linear(hidden_dim, hidden_dim)
        self.value_fc = nn.Linear(hidden_dim, hidden_dim)

        # additional linear layers for additive attention mechanism
        self.query_transform = nn.Linear(self.head_dim, self.head_dim)
        self.key_transform = nn.Linear(self.head_dim, self.head_dim)
        self.attention_score = nn.Linear(self.head_dim, 1)

        self.output_fc = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        # query = [batch size, query len, hidden dim]
        # key = [batch size, key len, hidden dim]
        # value = [batch size, value len, hidden dim]

        batch_size = query.shape[0]

        # apply linear transformations to query, key, and value
        Q = self.query_fc(query)
        K = self.key_fc(key)
        V = self.value_fc(value)
        # Q = K = V: [batch_size, src len, hidden_dim]

        # reshape and permute for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # Q = [batch_size, num_heads, query len, head_dim]

        # Additive Attention
        query_len = query.shape[1]
        transformed_query = self.query_transform(Q).view(
            batch_size, self.num_heads, query_len, 1, self.head_dim
        )

        key_len = key.shape[1]
        transformed_key = self.key_transform(K).view(
            batch_size, self.num_heads, 1, key_len, self.head_dim
        )

        # energy = [batch_size, num_heads, query len, key len]
        energy = torch.tanh(transformed_query + transformed_key)
        energy = self.attention_score(energy).squeeze(-1)

        # apply mask to ignore padding tokens
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # attention_weights = [batch_size, num_heads, query len, key len]
        attention_weights = torch.softmax(energy, dim=-1)

        # apply attention weights to the value
        # weighted_values = [batch_size, num_heads, query len, head_dim]
        weighted_values = torch.matmul(self.dropout(attention_weights), V)

        # reshape and concatenate heads
        weighted_values = weighted_values.permute(0, 2, 1, 3).contiguous()
        # weighted_values = [batch_size, query len, num_heads, head_dim]

        weighted_values = weighted_values.view(batch_size, -1, self.hidden_dim)
        # weighted_values = [batch_size, query len, hidden_dim]

        # apply final linear transformation
        output = self.output_fc(weighted_values)
        # output = [batch_size, query len, hidden_dim]

        return output, attention_weights


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, src len, hid dim]
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = AdditiveAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = AdditiveAttentionLayer(
            hid_dim, n_heads, dropout, device
        )
        self.feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        # trg = [batch_size, trg len, hid dim]

        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        # trg = [batch_size, trg len, hid dim]
        # attention = [batch_size, n heads, trg len, src len]

        _trg = self.feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        # trg = [batch_size, trg len, hid dim]

        return trg, attention


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        hid_dim,
        n_layers,
        n_heads,
        pf_dim,
        dropout,
        device,
        max_length=500,
    ):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                for _ in range(n_layers)
            ]
        )
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = (
            torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        )
        # pos: [batch_size, trg len]

        trg = self.dropout(
            (self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos)
        )
        # trg: [batch_size, trg len, hid dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        # trg: [batch_size, trg len, hid dim]
        # attention: [batch_size, n heads, trg len, src len]

        output = self.fc_out(trg)
        # output = [batch_size, trg len, output_dim]

        return output, attention
    

class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):

        #src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        #src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):

        #trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        #trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        #trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask
        #trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):

        #src = [batch size, src len]
        #trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)
        #enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]

        return output, attention
