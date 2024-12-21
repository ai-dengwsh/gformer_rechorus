import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.W_o(context)
        return output, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class GFormerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, start_idx=0):
        return self.pe[:, start_idx:start_idx + x.size(1)]

class GFormer(nn.Module):
    def __init__(self, user_count, item_count, d_model=256, num_heads=4, num_layers=2, d_ff=1024, dropout=0.1):
        super().__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.d_model = d_model
        
        # Embeddings
        self.user_embedding = nn.Embedding(user_count, d_model)
        self.item_embedding = nn.Embedding(item_count, d_model)
        
        # Position Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Layers
        self.layers = nn.ModuleList([
            GFormerLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.user_projection = nn.Linear(d_model, d_model)
        self.item_projection = nn.Linear(d_model, d_model)
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, inputs):
        users = inputs['user']
        items = inputs['item']
        
        # Get embeddings
        user_emb = self.user_embedding(users).unsqueeze(1)  # Add sequence dimension
        item_emb = self.item_embedding(items).unsqueeze(1)  # Add sequence dimension
        
        # Add position encoding
        user_emb = user_emb + self.pos_encoder(user_emb)
        item_emb = item_emb + self.pos_encoder(item_emb)
        
        # Process through transformer layers
        for layer in self.layers:
            user_emb = layer(user_emb)
            item_emb = layer(item_emb)
        
        # Remove sequence dimension and project embeddings
        user_output = self.user_projection(user_emb.squeeze(1))
        item_output = self.item_projection(item_emb.squeeze(1))
        
        return user_output, item_output
    
    @torch.no_grad()
    def get_embedding(self):
        # Get all user and item embeddings
        all_users = torch.arange(self.user_count).to(next(self.parameters()).device)
        all_items = torch.arange(self.item_count).to(next(self.parameters()).device)
        
        user_emb = self.user_embedding(all_users).unsqueeze(1)
        item_emb = self.item_embedding(all_items).unsqueeze(1)
        
        # Add position encoding
        user_emb = user_emb + self.pos_encoder(user_emb)
        item_emb = item_emb + self.pos_encoder(item_emb)
        
        # Process through transformer layers
        for layer in self.layers:
            user_emb = layer(user_emb)
            item_emb = layer(item_emb)
        
        # Remove sequence dimension and project embeddings
        user_output = self.user_projection(user_emb.squeeze(1))
        item_output = self.item_projection(item_emb.squeeze(1))
        
        return user_output, user_emb.squeeze(1), item_output, item_emb.squeeze(1)
    
    def get_loss(self, output):
        user_output, item_output = output
        
        # Normalize embeddings
        user_output = F.normalize(user_output, dim=-1)
        item_output = F.normalize(item_output, dim=-1)
        
        # Compute similarity scores
        logits = torch.matmul(user_output, item_output.transpose(-2, -1))
        
        # Create labels (diagonal matrix for positive pairs)
        labels = torch.eye(user_output.size(0)).to(logits.device)
        
        # Compute cross entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss 