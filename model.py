''' Model implementations should be placed here. '''

import os
import torch
import torch.nn as nn

from typing import Any, Dict

IN_KERNEL = os.environ.get('KAGGLE_WORKING_DIR') is not None

if not IN_KERNEL:
    from pytorchcv.model_provider import get_model
else:
    from model_provider import get_model


class ClassifierModel(nn.Module):
    ''' Just an image classifier. '''
    def __init__(self, config: Any, pretrained: bool) -> None:
        super().__init__()
        self.model = create_classifier_model(config, pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # type: ignore
        return self.model(x)

class SiameseModel(nn.Module):
    ''' Model with two inputs. '''
    def __init__(self, config: Any, pretrained: bool) -> None:
        super().__init__()
        self.head = create_classifier_model(config, pretrained)
        self.dropout = nn.Dropout(config.model.dropout) if config.model.dropout else None
        self.fc = nn.Linear(self.head.output.in_features, config.model.num_classes)
        self.num_channels = config.model.num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor: # type: ignore
        y1 = self.head.features(x[:, :self.num_channels])
        y2 = self.head.features(x[:, self.num_channels:])

        d = y1 - y2
        y = d.view(d.size(0), -1)

        if self.dropout is not None:
            y = self.dropout(y)

        y = self.fc(y)
        return y

def create_classifier_model(config: Any, pretrained: bool) -> Any:
    num_channels = config.model.num_channels

    if not IN_KERNEL:
        model = get_model(config.model.arch, pretrained=pretrained)
    else:
        model = get_model(config.model.arch, pretrained=pretrained, root='../input/pytorchcv-models/')

    # print(model)
    if num_channels != 3:
        # block = model.features[0].conv # for ResNet
        block = model.features[0] # for DenseNet
        block.conv = nn.Conv2d(in_channels=num_channels,
                               out_channels=block.conv.out_channels,
                               kernel_size=block.conv.kernel_size,
                               stride=block.conv.stride,
                               padding=block.conv.padding,
                               dilation=block.conv.dilation,
                               groups=block.conv.groups,
                               bias=block.conv.bias,
                               padding_mode=block.conv.padding_mode)

    if config.model.arch == 'xception':
        model.features[-1].pool = nn.AdaptiveAvgPool2d(1)
    else:
        model.features[-1] = nn.AdaptiveAvgPool2d(1)

    dropout = config.model.dropout

    if config.model.arch == 'pnasnet5large':
        if dropout == 0.0:
            model.output = nn.Linear(model.output[-1].in_features, config.model.num_classes)
        else:
            model.output = nn.Sequential(
                 nn.Dropout(dropout),
                 nn.Linear(model.output[-1].in_features, config.model.num_classes))
    elif config.model.arch == 'xception':
        if dropout < 0.1:
            model.output = nn.Linear(2048, config.model.num_classes)
        else:
            model.output = nn.Sequential(
                 nn.Dropout(dropout),
                 nn.Linear(2048, config.model.num_classes))
    elif config.model.arch.startswith('inception'):
        if dropout < 0.1:
            model.output = nn.Linear(model.output[-1].in_features, config.model.num_classes)
        else:
            model.output = nn.Sequential(
                 nn.Dropout(dropout),
                 nn.Linear(model.output[-1].in_features, config.model.num_classes))
    else:
        if dropout < 0.1:
            model.output = nn.Linear(model.output.in_features, config.model.num_classes)
        else:
            model.output = nn.Sequential(
                 nn.Dropout(dropout),
                 nn.Linear(model.output.in_features, config.model.num_classes))

    return model

def create_model(config: Any, pretrained: bool) -> Any:
    return globals()[config.model.type](config, pretrained)

def freeze_layers(model: Any) -> None:
    ''' Freezes all layers but the last FC layer. '''
    m = model.module
    for layer in m.children():
        for param in layer.parameters():
            param.requires_grad = False

    for layer in model.module.output.children():
        for param in layer.parameters():
            param.requires_grad = True

def unfreeze_layers(model: Any) -> None:
    ''' Unfreezes all layers. '''
    for layer in model.module.children():
        for param in layer.parameters():
            param.requires_grad = True
