##textcnn 网络搭建
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, pretrained_embedding, num_classes, pad_id=0):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(
            pretrained_embedding,
            freeze=False,
            padding_idx=pad_id
        )

        embed_dim = pretrained_embedding.size(1)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, 128, (2, embed_dim)),
            nn.Conv2d(1, 128, (3, embed_dim)),
            nn.Conv2d(1, 128, (4, embed_dim))
        ])

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 3, num_classes)

    def forward(self, x):
        x = self.embedding(x)

        x = x.unsqueeze(1)

        conv_outputs = []

        for conv in self.convs:
            out = conv(x)
            out = F.relu(out)
            out = out.squeeze(3)
            out = F.max_pool1d(out, out.size(2))
            out = out.squeeze(2)
            conv_outputs.append(out)

        out = torch.cat(conv_outputs, dim=1)
        out = self.dropout(out)
        out = self.fc(out)

        return out