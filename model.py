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

class SiameseWithNegControls(nn.Module):
    ''' Model with two inputs. '''
    def __init__(self, config: Any, pretrained: bool) -> None:
        super().__init__()

        self.num_channels = config.model.num_channels
        self.fc_layer_width = config.model.fc_layer_width
        self.combine_method = config.model.combine_method

        self.head = create_classifier_model(config, pretrained)

        num_inputs = self.head.output[-1].in_features
        if self.combine_method == 'concat':
            num_inputs *= 2
        elif self.combine_method == 'mpiotte':
            num_inputs *= 4
        elif self.combine_method == 'mpiotte_orig':
            self.conv1d_channels = config.model.conv1d_channels
            self.conv1 = nn.Conv1d(4, self.conv1d_channels, 1)
            self.conv2 = nn.Conv1d(self.conv1d_channels, 1, 1)

        self.fc1 = nn.Linear(num_inputs, self.fc_layer_width)
        self.batchnorm = nn.BatchNorm1d(self.fc_layer_width)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.model.dropout) if config.model.dropout else None
        self.fc2 = nn.Linear(config.model.fc_layer_width, config.model.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # type: ignore
        y1 = self.head.features(x[:, :self.num_channels])
        y2 = self.head.features(x[:, self.num_channels:])

        y1 = y1.view(y1.size(0), -1)
        y2 = y2.view(y2.size(0), -1)

        if self.combine_method == 'subtract':
            y = y1 - y2
        elif self.combine_method == 'concat':
            y = torch.cat([y1, y2], dim=1)
        elif self.combine_method == 'mpiotte':
            d = y1 - y2
            y = torch.cat([y1 + y2, y1 * y2, torch.abs(d), d * d], dim=1)
        elif self.combine_method == 'mpiotte_orig':
            y1 = y1.view(y1.size(0), 1, -1)
            y2 = y2.view(y2.size(0), 1, -1)

            d = y1 - y2
            y = torch.cat([y1 + y2, y1 * y2, torch.abs(d), d * d], dim=1)
            y = self.conv1(y)
            y = self.relu(y)
            y = self.conv2(y)

            y = y.view(y.size(0), -1)
        else:
            assert False

        y = self.fc1(y)
        y = self.batchnorm(y)
        y = self.relu(y)

        if self.dropout is not None:
            y = self.dropout(y)

        y = self.fc2(y)
        return y

class SiameseBinaryClassifier(nn.Module):
    ''' A binary classifier. Takes two inputs, returns boolean result:
    same class / not same class. '''
    def __init__(self, config: Any, pretrained: bool) -> None:
        super().__init__()

        self.num_channels = config.model.num_channels
        self.fc_layer_width = config.model.fc_layer_width
        self.combine_method = config.model.combine_method

        self.head = create_classifier_model(config, pretrained)

        num_inputs = self.head.output[-1].in_features
        if self.combine_method == 'concat':
            num_inputs *= 2
        elif self.combine_method == 'mpiotte':
            num_inputs *= 4
        elif self.combine_method == 'mpiotte_orig':
            self.conv1d_channels = config.model.conv1d_channels
            self.conv1 = nn.Conv1d(4, self.conv1d_channels, 1)
            self.conv2 = nn.Conv1d(self.conv1d_channels, 1, 1)

        if self.fc_layer_width:
            self.fc1 = nn.Linear(num_inputs, self.fc_layer_width)
            self.batchnorm = nn.BatchNorm1d(self.fc_layer_width)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(config.model.dropout) if config.model.dropout else None

        self.fc2 = nn.Linear(config.model.fc_layer_width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # type: ignore
        y1 = self.head.features(x[:, :self.num_channels])
        y2 = self.head.features(x[:, self.num_channels:])

        y1 = y1.view(y1.size(0), -1)
        y2 = y2.view(y2.size(0), -1)

        if self.combine_method == 'subtract':
            y = y1 - y2
        elif self.combine_method == 'concat':
            y = torch.cat([y1, y2], dim=1)
        elif self.combine_method == 'mpiotte':
            d = y1 - y2
            y = torch.cat([y1 + y2, y1 * y2, torch.abs(d), d * d], dim=1)
        elif self.combine_method == 'mpiotte_orig':
            y1 = y1.view(y1.size(0), 1, -1)
            y2 = y2.view(y2.size(0), 1, -1)

            d = y1 - y2
            y = torch.cat([y1 + y2, y1 * y2, torch.abs(d), d * d], dim=1)
            y = self.conv1(y)
            y = self.relu(y)
            y = self.conv2(y)

            y = y.view(y.size(0), -1)
        else:
            assert False

        if self.fc_layer_width:
            y = self.fc1(y)
            y = self.batchnorm(y)
            y = self.relu(y)

            if self.dropout is not None:
                y = self.dropout(y)

        y = self.fc2(y)
        return y

def create_classifier_model(config: Any, pretrained: bool) -> Any:
    num_channels = config.model.num_channels

    if not IN_KERNEL:
        model = get_model(config.model.arch, pretrained=pretrained)
    else:
        model = get_model(config.model.arch, pretrained=pretrained, root='../input/pytorchcv-models/')

    if num_channels != 3:
        if config.model.arch.startswith('resnet'):
            block = model.features[0].conv
        elif config.model.arch.startswith('densenet'):
            block = model.features[0]
        else:
            assert False

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
    return globals()[config.model.name](config, pretrained)

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
