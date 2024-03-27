import torch.nn as nn


class singlePeriodNet(nn.Module):
    def __init__(self, sizes, activation, output_activation, dropout, use_batchnorm):
        super(singlePeriodNet, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(x, y) for x, y in zip(sizes[:-1], sizes[1:])])
        if use_batchnorm:
            self.batchnorm1 = nn.BatchNorm1d(sizes[1])
            self.batchnorm2 = nn.BatchNorm1d(sizes[-2])
        for layer in self.linears:
            nn.init.xavier_normal_(layer.weight.data, gain=1.5)
        self.activation = activation
        if output_activation:
            self.output_activation = output_activation
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout

    def forward(self, x):
        x = x.T
        x = self.linears[0](x)
        x = self.activation(x)
        if self.use_batchnorm:
            x = self.batchnorm1(x)
        x = nn.Dropout(self.dropout)(x)
        # Skip connection 1/2
        y = x
        for layer in self.linears[1:-2]:
            x = layer(x)
            x = self.activation(x)
            x = nn.Dropout(self.dropout)(x)
        # Skip connection 2/2
        x = x + y
        x = self.linears[-2](x)
        x = self.activation(x)
        if self.use_batchnorm:
            x = self.batchnorm2(x)
        x = nn.Dropout(self.dropout)(x)
        x = self.linears[-1](x)
        if self.output_activation:
            x = self.output_activation(x)
        return x.T

    def freeze(self, parameters=None):
        if parameters:
            for parameter in parameters:
                parameter.requires_grad = False
        else:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def unfreeze(self, parameters=None):
        if parameters:
            for parameter in parameters:
                parameter.requires_grad = True
        else:
            for parameter in self.parameters():
                parameter.requires_grad = True


class timegridNet(nn.Module):
    def __init__(self, sizes, n_maturities, activation, output_activation, dropout, use_batchnorm):
        super(timegridNet, self).__init__()
        self.singlePeriodNets = nn.ModuleList([singlePeriodNet(sizes, activation, output_activation, dropout, use_batchnorm) for _ in range(n_maturities)])

    def forward(self, idx, x):
        x = self.singlePeriodNets[idx](x)
        return x

    def freeze(self, parameters=None):
        if parameters:
            for parameter in self.singlePeriodNets.parameters():
                parameter.requires_grad = False
        else:
            for parameter in self.parameters():
                parameter.requires_grad = False

    def unfreeze(self, parameters=None):
        if parameters:
            for parameter in parameters:
                parameter.requires_grad = True
        else:
            for parameter in self.parameters():
                parameter.requires_grad = True
