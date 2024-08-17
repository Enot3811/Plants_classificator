import sys
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models.efficientnet as efficientnet
import torchvision.models.resnet as resnet

sys.path.append(str(Path(__file__).parents[1]))
from utils.torch_utils.torch_modules import Flatten


class ModelFactory:

    @staticmethod
    def create_model_by_name(name: str, num_classes: int) -> nn.Module:
        if name == 'resnext101_64x4d':
            model = resnet.resnext101_64x4d(
                weights=resnet.ResNeXt101_64X4D_Weights.DEFAULT)
            layers = list(model.children())[:-1]
            layers.append(Flatten())
            layers.append(nn.Linear(2048, out_features=num_classes))
            model = nn.Sequential(*layers)
        elif name == 'efficientnet_b4':
            model = efficientnet.efficientnet_b4(
                weights=efficientnet.EfficientNet_B4_Weights.DEFAULT)
            layers = list(model.children())[:-1]
            layers.append(nn.Sequential(*[
                Flatten(),
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(1792, out_features=num_classes)
            ]))
            model = torch.nn.Sequential(*layers)
        else:
            raise NotImplementedError(f'{name} is not implemented.')
        return model


if __name__ == '__main__':
    name = 'efficientnet_b4'
    # name = 'resnext101_64x4d'
    model = ModelFactory.create_model_by_name(name, 3000)
    print(model)
    input = torch.randn((1, 3, 224, 224))
    model.eval()
    out = model(input)
    print(out.shape)
