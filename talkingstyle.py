import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from hubert import HubertModel

# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

# Alignment Bias
def enc_dec_mask(device, dataset, T, S):
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    return (mask==1).to(device=device)

# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(Attention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        self.Wq = nn.Linear(1, embed_dim)
        self.Wk = nn.Linear(1, embed_dim)
        self.Wv = nn.Linear(1, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1)


    def forward(self, obj_vector, my_obj_vector):

        obj_vector = obj_vector.unsqueeze(-1)
        my_obj_vector = my_obj_vector.unsqueeze(-1)

        Q = self.Wq(obj_vector)
        K = self.Wk(my_obj_vector)
        V = self.Wv(my_obj_vector)

        # 将obj_vector作为查询张量，my_obj_vector作为键值对张量
        attention_output, _ = self.attention(Q, K, V)

        attention_output = self.head(self.norm(attention_output)).squeeze(-1)

        return attention_output

# 定义残差连接
class ResidualConnect(nn.Module):
    def __init__(self):
        super(ResidualConnect, self).__init__()
    def forward(self, inputs, residual):
        return F.relu(inputs + residual)
    

class Style_Encoder(nn.Module):
    def __init__(self, args, deep=4):
        super(Style_Encoder, self).__init__()
        self.obj_vector = nn.Linear(len(args.train_subjects.split()), args.feature_dim, bias=False)
        self.my_obj_vector = nn.Embedding(len(args.train_subjects.split()), args.feature_dim)
        
        # 构建i层的注意力机制
        self.dropout = nn.Dropout(0.1)
        self.attention = Attention(4, num_heads=4)

        self.residual_connect = ResidualConnect()

    def forward(self, one_hot):
        obj_embedding = self.obj_vector(one_hot)#(1, feature_dim)
        my_obj_embedding = self.my_obj_vector(torch.argmax(one_hot, dim=1))
        attention_output = self.dropout(self.attention(obj_embedding, my_obj_embedding))
        obj_embedding = self.residual_connect(obj_embedding, attention_output)

        return obj_embedding


class TalkingStyle(nn.Module):
    def __init__(self, args, mouth_loss_ratio=0.6):
        super(TalkingStyle, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.dataset = args.dataset
        self.device = args.device
        self.audio_encoder = HubertModel.from_pretrained("/workspace/hubert-large-ls960-ft")
        # hubert weights initialization
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_feature_map = nn.Linear(1024, args.feature_dim)
        # motion encoder
        self.vertice_encoder = nn.Linear(args.vertice_dim, args.feature_dim)
        # periodic positional encoding 
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period = args.period)
        # temporal bias
        self.biased_mask = init_biased_mask(n_head = 4, max_seq_len = 600, period=args.period)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=4, dim_feedforward=2*args.feature_dim, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        # motion decoder
        self.vertice_decoder = nn.Linear(args.feature_dim, args.vertice_dim)
        # style encoder
        self.style_encoder = Style_Encoder(args)
        nn.init.constant_(self.vertice_decoder.weight, 0)
        nn.init.constant_(self.vertice_decoder.bias, 0)
        # load mouth region
        with open(args.lip_region) as f:
            maps = f.read().split(", ")
            self.mouth_map = [int(i) for i in maps]
        self.mouth_loss_ratio = mouth_loss_ratio

    def forward(self, audio, template, vertice, one_hot, criterion, teacher_forcing=True):
        # tgt_mask: :math:`(T, T)`.
        # memory_mask: :math:`(T, S)`.
        template = template.unsqueeze(1) # (1,1, V*3)
        # encoder the style
        style_emb = self.style_encoder(one_hot)

        frame_num = vertice.shape[1]
        hidden_states = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state
        if self.dataset == "BIWI":
            if hidden_states.shape[1]<frame_num*2:
                vertice = vertice[:, :hidden_states.shape[1]//2]
                frame_num = hidden_states.shape[1]//2
        hidden_states = self.audio_feature_map(hidden_states)

        if teacher_forcing:
            vertice_input = torch.cat((template, vertice[:,:-1]), 1) # shift one position
            vertice_input = vertice_input - template
            vertice_input = self.vertice_encoder(vertice_input)
            vertice_input = vertice_input + style_emb
            vertice_input = self.PPE(vertice_input)
            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            vertice_out = self.vertice_decoder(vertice_out)
        else:
            for i in range(frame_num):
                if i==0:
                    vertice_emb = style_emb.unsqueeze(0) # (1,1,feature_dim)
                    vertice_input = self.PPE(vertice_emb)
                else:
                    vertice_input = self.PPE(vertice_emb)
                tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
                memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
                vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
                vertice_out = self.vertice_decoder(vertice_out)
                new_output = self.vertice_encoder(vertice_out[:,-1,:]).unsqueeze(1)
                new_output = new_output + style_emb
                vertice_emb = torch.cat((vertice_emb, new_output), 1)

        vertice_out = vertice_out + template
        all_loss = criterion(vertice_out, vertice) # (batch, seq_len, V*3)
        mouth_loss = criterion(vertice_out[:, :, self.mouth_map], vertice[:, :, self.mouth_map])
        loss = (1-self.mouth_loss_ratio) * all_loss + self.mouth_loss_ratio * mouth_loss
        return loss, [all_loss, mouth_loss]

    def predict(self, audio, template, one_hot):
        template = template.unsqueeze(1) # (1,1, V*3)

        style_emb = self.style_encoder(one_hot)
        
        hidden_states = self.audio_encoder(audio, self.dataset).last_hidden_state
        if self.dataset == "BIWI":
            frame_num = hidden_states.shape[1]//2
        elif self.dataset == "vocaset":
            frame_num = hidden_states.shape[1]
        hidden_states = self.audio_feature_map(hidden_states)

        for i in range(frame_num):
            if i==0:
                vertice_emb = style_emb.unsqueeze(0)
                vertice_input = self.PPE(vertice_emb)
            else:
                vertice_input = self.PPE(vertice_emb)

            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            vertice_out = self.vertice_decoder(vertice_out)

            new_output = self.vertice_encoder(vertice_out[:,-1,:]).unsqueeze(1)
            new_output = new_output + style_emb
            vertice_emb = torch.cat((vertice_emb, new_output), 1)

        vertice_out = vertice_out + template
        return vertice_out
