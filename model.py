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


class SiameseModel(nn.Module):
    ''' Model with two inputs. '''
    def __init__(self, config: Any, pretrained: bool) -> None:
        super().__init__()
        self.head = create_model_head(config, pretrained)
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

class SiameseModel2(nn.Module):
    ''' Model with two inputs. '''
    def __init__(self, config: Any, pretrained: bool) -> None:
        super().__init__()
        self.head = create_model_head(config, pretrained)
        self.dropout = nn.Dropout(config.model.dropout) if config.model.dropout else None
        self.fc = nn.Linear(2 * self.head.output.in_features, config.model.num_classes)
        self.num_channels = config.model.num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor: # type: ignore
        y1 = self.head.features(x[:, :self.num_channels])
        y2 = self.head.features(x[:, self.num_channels:])

        y = torch.cat([y1, y2], dim=1)
        y = y.view(y.size(0), -1)

        if self.dropout is not None:
            y = self.dropout(y)

        y = self.fc(y)
        return y

class SiameseModel3(nn.Module):
    ''' Model with two inputs. '''
    def __init__(self, config: Any, pretrained: bool) -> None:
        super().__init__()
        self.head = create_model_head(config, pretrained)
        self.dropout = nn.Dropout(config.model.dropout) if config.model.dropout else None
        self.num_hidden = config.model.num_hidden
        self.fc = nn.Linear(2 * self.head.output.in_features, self.num_hidden)
        self.fc2 = nn.Linear(self.num_hidden, config.model.num_classes)
        self.num_channels = config.model.num_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor: # type: ignore
        y1 = self.head.features(x[:, :self.num_channels])
        y2 = self.head.features(x[:, self.num_channels:])

        y = torch.cat([y1, y2], dim=1)
        y = y.view(y.size(0), -1)

        if self.dropout is not None:
            y = self.dropout(y)

        y = self.fc(y)

        if self.dropout is not None:
            y = self.dropout(y)

        y = self.fc2(y)
        return y


def create_model_head(config: Any, pretrained: bool) -> Any:
    if not IN_KERNEL:
        model = get_model(config.model.arch, pretrained=pretrained)
    else:
        model = get_model(config.model.arch, pretrained=pretrained, root='../input/pytorchcv-models/')

    if config.model.arch == 'xception':
        model.features[-1].pool = nn.AdaptiveAvgPool2d(1)
    else:
        model.features[-1] = nn.AdaptiveAvgPool2d(1)

    # dropout = config.model.dropout
    #
    # if config.model.arch == 'pnasnet5large':
    #     if dropout == 0.0:
    #         model.output = nn.Linear(model.output[-1].in_features, config.model.num_classes)
    #     else:
    #         model.output = nn.Sequential(
    #              nn.Dropout(dropout),
    #              nn.Linear(model.output[-1].in_features, config.model.num_classes))
    # elif config.model.arch == 'xception':
    #     if dropout < 0.1:
    #         model.output = nn.Linear(2048, config.model.num_classes)
    #     else:
    #         model.output = nn.Sequential(
    #              nn.Dropout(dropout),
    #              nn.Linear(2048, config.model.num_classes))
    # elif config.model.arch.startswith('inception'):
    #     if dropout < 0.1:
    #         model.output = nn.Linear(model.output[-1].in_features, config.model.num_classes)
    #     else:
    #         model.output = nn.Sequential(
    #              nn.Dropout(dropout),
    #              nn.Linear(model.output[-1].in_features, config.model.num_classes))
    # else:
    #     if dropout < 0.1:
    #         model.output = nn.Linear(model.output.in_features, config.model.num_classes)
    #     else:
    #         model.output = nn.Sequential(
    #              nn.Dropout(dropout),
    #              nn.Linear(model.output.in_features, config.model.num_classes))

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
