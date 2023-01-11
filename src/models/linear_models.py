import torch


def create_model(name, input_dim, num_classes=1, pretrained=False):
    assert not pretrained, ValueError('No support for pretrained linear models')

    if name == 'no_bias':
        bias = False
    else:
        bias = True

    return torch.nn.Linear(input_dim, num_classes, bias=bias)
