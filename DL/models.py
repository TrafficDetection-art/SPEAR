import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, max_len):
        super(CNNLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.conv1 = nn.Conv1d(embed_size, 128, 5, padding=2)
        self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(128 * 2, num_classes)

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
    def __init__(self, vocab_size, embed_size, num_classes, max_len):
        super(TextCNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.conv1 = nn.Conv2d(1, 128, (3, embed_size))
        self.conv2 = nn.Conv2d(1, 128, (4, embed_size))
        self.conv3 = nn.Conv2d(1, 128, (5, embed_size))
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(128 * 3, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids).unsqueeze(1)
        x1 = F.relu(self.conv1(x)).squeeze(3)
        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)
        x2 = F.relu(self.conv2(x)).squeeze(3)
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)
        x3 = F.relu(self.conv3(x)).squeeze(3)
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)
        x = torch.cat((x1, x2, x3), 1)
        x = self.drop(x)
        return self.fc(x)

class DNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, max_len):
        super(DNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc1 = nn.Linear(embed_size * max_len, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        return self.fc3(x)

class DeepLog(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, max_len, num_layers=8):
        super(DeepLog, self).__init__()
        self.hidden_size = 512
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, self.hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
