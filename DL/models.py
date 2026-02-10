import sys; sys.path.insert(0, '..')
from project_settings import get

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, max_len,
                 conv_out_channels=None, conv_kernel_size=None, conv_padding=None,
                 lstm_hidden_size=None, dropout=None):
        super().__init__()
        arch = get("dl.model_architectures.cnn_lstm", {})
        conv_out = conv_out_channels or arch.get("conv_out_channels", 128)
        kernel = conv_kernel_size or arch.get("conv_kernel_size", 5)
        pad = conv_padding or arch.get("conv_padding", 2)
        lstm_h = lstm_hidden_size or arch.get("lstm_hidden_size", 128)
        drop_p = dropout or arch.get("dropout", 0.3)
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.conv1 = nn.Conv1d(embed_size, conv_out, kernel, padding=pad)
        self.lstm = nn.LSTM(conv_out, lstm_h, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p=drop_p)
        self.fc = nn.Linear(lstm_h * 2, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = x.permute(0, 2, 1)
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        output = self.drop(hidden)
        return self.fc(output)


class TextCNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, max_len,
                 num_filters=None, kernel_sizes=None, dropout=None):
        super().__init__()
        arch = get("dl.model_architectures.textcnn", {})
        nf = num_filters or arch.get("num_filters", 128)
        ks = kernel_sizes or arch.get("kernel_sizes", [3, 4, 5])
        drop_p = dropout or arch.get("dropout", 0.5)
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, nf, (k, embed_size)) for k in ks
        ])
        self.drop = nn.Dropout(p=drop_p)
        self.fc = nn.Linear(nf * len(ks), num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids).unsqueeze(1)
        conv_results = []
        for conv in self.convs:
            c = F.relu(conv(x)).squeeze(3)
            c = F.max_pool1d(c, c.size(2)).squeeze(2)
            conv_results.append(c)
        x = torch.cat(conv_results, 1)
        x = self.drop(x)
        return self.fc(x)


class DNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, max_len,
                 hidden_sizes=None, dropout=None):
        super().__init__()
        arch = get("dl.model_architectures.dnn", {})
        hs = hidden_sizes or arch.get("hidden_sizes", [512, 256])
        drop_p = dropout or arch.get("dropout", 0.5)
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc1 = nn.Linear(embed_size * max_len, hs[0])
        self.fc2 = nn.Linear(hs[0], hs[1] if len(hs) > 1 else num_classes)
        self.fc3 = nn.Linear(hs[1] if len(hs) > 1 else hs[0], num_classes)
        self.drop = nn.Dropout(p=drop_p)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        return self.fc3(x)


class DeepLog(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, max_len,
                 hidden_size=None, num_layers=None):
        super().__init__()
        arch = get("dl.model_architectures.deeplog", {})
        self.hidden_size = hidden_size or arch.get("hidden_size", 512)
        self.num_layers = num_layers or arch.get("num_layers", 8)
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
