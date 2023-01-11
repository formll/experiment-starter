"""Code source: https://github.com/meliketoy/wide-resnet.pytorch"""
from .lenet import *
from .vggnet import *
from .resnet import *
from .wideresnet import *


def _default_cfg():
    return dict(input_size=(3, 32, 32),
                mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010),
                interpolation='bicubic',  # ?
                crop_pct=1.0,
                )


def create_model(name, pretrained=False, num_classes=10):
    assert not pretrained, ValueError('"Tiny models" do not come pretrained checkpoints')

    name, *args = name.split('_')
    if name == 'lenet':
        model = LeNet(num_classes)
    elif name == 'resnet':
        depth = int(args[0])
        model = ResNet(depth, num_classes)
    elif name == 'vgg':
        depth = int(args[0])
        model = VGG(depth, num_classes)
    elif name == 'wrn':
        depth, widen = int(args[0]), int(args[1])
        if len(args) > 2:
            dropout = float(args[2])
        else:
            dropout = 0.0
        model = WideResNet(depth, widen, dropout, num_classes)
    else:
        raise ValueError(f'Model name "{name}" not found in the "tiny models" module')

    model.default_cfg = _default_cfg()

    return model
