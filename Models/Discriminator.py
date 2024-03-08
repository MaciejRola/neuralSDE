import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, N_steps, num_layers, layer_size, device):
        super(Discriminator, self).__init__()
        sizes = [N_steps + 1] + [layer_size] * num_layers + [1]
        self.N_steps = N_steps
        self.layers = nn.ModuleList([nn.Linear(x, y) for x, y in zip(sizes[:-1], sizes[1:])])
        self.device = device
        self.activation_function = nn.ReLU()
        self.output_activation_function = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation_function(layer(x))
        return self.layers[-1](x)

    def predict(self, x):
        return bool(round(self.forward(x)))