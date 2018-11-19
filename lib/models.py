import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from cnf.APConfig import mconfig


def save_model(filepath, model, user_info={}):
    info = {
        "net": model.state_dict(),
        "user_info": user_info
    }
    torch.save(info, filepath)


def load_model(filepath, model):
    info = torch.load(filepath, map_location=lambda storage, loc: storage)
    model.load_state_dict(info["net"])
    return info


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight, bias)


class SimpleFFDQN(nn.Module):
    tensor_width: int

    def __init__(self, obs_len, actions_n, tensor_width=mconfig.tensor_width):
        super(SimpleFFDQN, self).__init__()

        self.tensor_width = tensor_width

        self.fc_val = nn.Sequential(
            nn.Linear(obs_len, self.tensor_width),
            nn.ReLU(),
            nn.Linear(self.tensor_width, self.tensor_width),
            nn.ReLU(),
            nn.Linear(self.tensor_width, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(obs_len, self.tensor_width),
            nn.ReLU(),
            nn.Linear(self.tensor_width, self.tensor_width),
            nn.ReLU(),
            nn.Linear(self.tensor_width, actions_n)
        )

    def forward(self, x):
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return val + adv - adv.mean()


class DQNConv1D(nn.Module):
    def __init__(self, shape, actions_n):
        super(DQNConv1D, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(shape[0], 256, 5),
            nn.ReLU(),
            nn.Conv1d(256, 256, 5),
            nn.ReLU(),
        )

        out_size = self._get_conv_out(shape)

        self.fc_val = nn.Sequential(
            nn.Linear(out_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(out_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, actions_n)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean()


# 마직 열은 Linear 로 바로 연결한다.
class DQNConv1DLastLinear(nn.Module):
    def __init__(self, shape, actions_n):
        super(DQNConv1DLastLinear, self).__init__()

        shape[0] -= 1
        kernel_size = shape[0]

        self.conv = nn.Sequential(
            nn.Conv1d(kernel_size, 256, kernel_size),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size),
            nn.ReLU(),
        )

        out_size = self._get_conv_out(shape)

        self.fc_val = nn.Sequential(
            nn.Linear(out_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(out_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, actions_n)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x[:-1]).view(x.size()[0]-1, -1)

        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean()

class DQNConv1DLarge(nn.Module):
    def __init__(self, shape, actions_n):
        super(DQNConv1DLarge, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(shape[0], 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.MaxPool1d(3, 2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3),
            nn.ReLU(),
        )

        out_size = self._get_conv_out(shape)

        self.fc_val = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(out_size, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean()
