import torch.nn as nn

from torch.nn.utils import weight_norm


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.first_conv = weight_norm(nn.Conv2d(in_channels, out_channels, (1, kernel_size), stride, dilation=dilation))
        self.second_conv = weight_norm(nn.Conv2d(out_channels, out_channels, (1, kernel_size), stride, dilation=dilation))

        self.network = nn.Sequential(
            nn.ZeroPad2d((padding, 0, 0, 0)),
            self.first_conv,
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.ZeroPad2d((padding, 0, 0, 0)),
            self.second_conv,
            nn.ReLU(),
            nn.Dropout(dropout))

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        self.first_conv.weight.data.normal_(0, 0.01)
        self.second_conv.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        output = self.network(x.unsqueeze(2)).squeeze(2)
        residual = x if self.downsample is None else self.downsample(x)

        return self.relu(output + residual)


class TemporalConvNet(nn.Module):
    def __init__(self, in_channels, out_channels_list, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()

        temporal_blocks = list()
        for index in range(0, len(out_channels_list)):
            dilation = 2 ** index
            in_channels = in_channels if index == 0 else out_channels_list[index - 1]
            out_channels = out_channels_list[index]

            temporal_blocks.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation,
                                                 padding=(kernel_size - 1) * dilation, dropout=dropout))

            self.network = nn.Sequential(*temporal_blocks)

    def forward(self, x):
        return self.network(x)
