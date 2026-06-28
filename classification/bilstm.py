##bilstm 网络搭建
import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, pretrained_embedding, hidden_dim, num_classes, pad_id=0):
        super(BiLSTM, self).__init__()

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

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]

        x = self.embedding(x)
        # x: [batch_size, seq_len, embed_dim]

        output, (hidden, cell) = self.lstm(x)
        # output: [batch_size, seq_len, hidden_dim * 2]

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        out = torch.cat((forward_hidden, backward_hidden), dim=1)
        # out: [batch_size, hidden_dim * 2]

        out = self.dropout(out)

        out = self.fc(out)
        # out: [batch_size, num_classes]

        return out