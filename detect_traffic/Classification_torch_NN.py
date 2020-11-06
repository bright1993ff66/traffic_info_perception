import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from Config.config import Configurable
from data_paths import random_seed, project_path

torch.manual_seed(random_seed)
random.seed(random_seed)


class CNN_LSTM(nn.Module):

    def __init__(self, args):
        super(CNN_LSTM, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        if args.THREE_CLASS_TASK:
            C = 3
        else:
            C = 2
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        self.C = C
        self.embed = nn.Embedding(V, D)
        # pretrained  embedding
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)

        # CNN
        self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.dropout = nn.Dropout(args.dropout)
        # for cnn cuda
        if self.args.cuda is True:
            for conv in self.convs1:
                conv = conv.cuda()

        # LSTM
        self.lstm = nn.LSTM(D, self.hidden_dim, dropout=args.dropout, num_layers=self.num_layers)

        # linear
        L = len(Ks) * Co + self.hidden_dim
        self.hidden2label1 = nn.Linear(L, L // 2)
        self.hidden2label2 = nn.Linear(L // 2, C)

    def forward(self, x):
        embed = self.embed(x)

        # CNN
        cnn_x = embed
        cnn_x = torch.transpose(cnn_x, 0, 1)
        cnn_x = cnn_x.unsqueeze(1)
        cnn_x = [F.relu(conv(cnn_x)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        cnn_x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cnn_x]  # [(N,Co), ...]*len(Ks)
        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = self.dropout(cnn_x)

        # LSTM
        lstm_x = embed.view(len(x), embed.size(1), -1)
        lstm_out, _ = self.lstm(lstm_x)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)

        # CNN and LSTM cat
        cnn_x = torch.transpose(cnn_x, 0, 1)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        cnn_lstm_out = torch.cat((cnn_x, lstm_out), 0)
        cnn_lstm_out = torch.transpose(cnn_lstm_out, 0, 1)

        # linear
        cnn_lstm_out = self.hidden2label1(F.tanh(cnn_lstm_out))
        cnn_lstm_out = self.hidden2label2(F.tanh(cnn_lstm_out))

        # output
        logit = cnn_lstm_out
        return logit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configure Traffic Weibo Detection Module...")
    parser.add_argument('--config_file', default=os.path.join(project_path, 'Config', 'config.cfg'))
    config = parser.parse_args()
    config = Configurable(config_file=config.config_file)
    model = CNN_LSTM(args=config)
    print(model)
