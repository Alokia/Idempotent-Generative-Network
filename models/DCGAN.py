import torch
from torch import nn


def get_activation(activation):
    if activation == 'none':
        return None
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError('Unknown activation function: {}'.format(activation))


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 norm=True, n_groups=32, activation='relu', transposed=False):
        super().__init__()
        if transposed:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(n_groups, in_channels) if norm else None
        self.activation = get_activation(activation)

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model_seq = nn.Sequential(
            BaseConv(in_channels, 64, 4, 2, 1, norm=False, activation='leaky_relu'),
            BaseConv(64, 128, 4, 2, 1, activation='leaky_relu'),
            BaseConv(128, 256, 4, 2, 1, activation='leaky_relu'),
            BaseConv(256, 512, 4, 2, 1, activation='leaky_relu'),
            BaseConv(512, 512, 4, 1, 0, norm=False, activation='none'),
        )

    def forward(self, x):
        x = self.model_seq(x)
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels=3):
        super().__init__()
        self.model_seq = nn.Sequential(
            BaseConv(512, 512, 4, 1, 0, transposed=True),
            BaseConv(512, 256, 4, 2, 1, transposed=True),
            BaseConv(256, 128, 4, 2, 1, transposed=True),
            BaseConv(128, 64, 4, 2, 1, transposed=True),
            BaseConv(64, out_channels, 4, 2, 1, norm=False, activation='tanh', transposed=True),
        )

    def forward(self, x):
        x = self.model_seq(x)
        return x


class DCGAN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(out_channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    x = torch.randn(3, 3, 64, 64)
    model = DCGAN(3, 3)
    out = model(x)
    print(out.shape)
