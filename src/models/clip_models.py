import pdb

import clip
import torch
from functools import partial


def _model_to_dim(name):
    if name.startswith('ViT-B'):
        return 512
    elif name.startswith('ViT-L'):
        return 768
    elif name.startswith('RN50'):
        return dict(RN50=1024, RN101=512, RN50x4=640, RN50x16=768, RN50x64=1024)[name]
    elif name == 'RN101':
        return 1024
    else:
        raise ValueError(f'Unknown architecture name {name}')


def _default_config():
    return dict(input_size=(3, 224, 224),
                mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711),
                interpolation='bicubic',  # ?
                crop_pct=1.0,
                )


class OpenAIClipWrapper(torch.nn.Module):
    def __init__(self, name, num_classes, normalize=False, initial_weights=None, cast_to_float=True):
        super(OpenAIClipWrapper, self).__init__()
        self.model, _ = clip.load(name, device='cpu')
        self.classification_head = torch.nn.Linear(_model_to_dim(name), num_classes)
        self.normalize = normalize

        if initial_weights is not None:
            self.classification_head.weight.copy_(initial_weights)
            self.classification_head.bias.zero_()

        # Get rid of the language part.
        delattr(self.model, 'transformer')

        if cast_to_float:
            for p in self.model.parameters():
                p.data = p.data.float()
                # if p.grad:
                #     p.grad.data = p.grad.data.float()

    def forward(self, images):
        features = self.model.encode_image(images)  # .float()
        if self.normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        return self.classification_head(features)

    def get_classifier(self):
        return self.classification_head


def create_model(name, pretrained=True, num_classes=1000):
    assert pretrained, ValueError('Does not support non-pretrained models')

    def reformat_name(in_name):
        # example valid inputs: vit_b32, vit_l14, vit_l14_336, resnet_50, resent_50_16
        in_name, *args = in_name.split('_')
        if in_name == 'vit':
            out_name = f'ViT-{args[0][0].upper()}/{args[0][1:]}'
            if len(args) > 1:
                out_name += f'@{args[1]}px'
        elif in_name == 'resnet':
            out_name = f'RN{args[0]}'
            if len(args) > 1:
                out_name += f'x{args[1]}'
        else:
            raise ValueError(f'Unknown CLIP architecture {in_name}')
        return out_name

    model = OpenAIClipWrapper(name=reformat_name(name), num_classes=num_classes)

    model.default_cfg = _default_config()
    model.num_classes = num_classes

    return model
