#bilstm_attention 网络搭建
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMAttention(nn.Module):
    def __init__(self, pretrained_embedding, hidden_dim, num_classes, pad_id=0):
        super(BiLSTMAttention, self).__init__()

        self.pad_id = pad_id

        self.embedding = nn.Embedding.from_pretrained(
            pretrained_embedding,
            freeze=False,
            padding_idx=pad_id
        )

        embed_dim = pretrained_embedding.size(1)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]

        embed = self.embedding(x)
        # embed: [batch_size, seq_len, embed_dim]

        output, (hidden, cell) = self.lstm(embed)
        # output: [batch_size, seq_len, hidden_dim * 2]

        score = self.attention(output)
        # score: [batch_size, seq_len, 1]

        score = score.squeeze(-1)
        # score: [batch_size, seq_len]

        mask = (x != self.pad_id)
        score = score.masked_fill(mask == 0, -1e9)

        attention_weight = F.softmax(score, dim=1)
        # attention_weight: [batch_size, seq_len]

        attention_weight = attention_weight.unsqueeze(1)
        # attention_weight: [batch_size, 1, seq_len]

        context = torch.bmm(attention_weight, output)
        # context: [batch_size, 1, hidden_dim * 2]

        context = context.squeeze(1)
        # context: [batch_size, hidden_dim * 2]

        context = self.dropout(context)

        out = self.fc(context)
        # out: [batch_size, num_classes]

        return out