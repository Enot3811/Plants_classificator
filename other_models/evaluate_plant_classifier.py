"""Скрипт для теста классификатора растений."""

from pathlib import Path
import json
import argparse
import sys
import csv

import torch
import torchvision
import albumentations as A
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[1]))
from dataset_utils.plants_dataset import PlantsDataset
    

class Flatten(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.flatten(x, 1)


def main(config_pth: Path):
    # Read config
    with open(config_pth, 'r') as f:
        config_str = f.read()
    config = json.loads(config_str)

    # Prepare some stuff
    torch.manual_seed(config['random_seed'])
    work_dir = Path(config['work_dir'])
    ckpt_dir = work_dir / 'ckpts'
    cls_to_id = work_dir / 'cls_to_id.csv'

    if config['device'] == 'auto':
        device: torch.device = (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu'))
    else:
        device: torch.device = torch.device(config['device'])

    # Get dataset
    with open(cls_to_id) as f:
        cls_to_id = dict(map(
            lambda cls_pair: (cls_pair[0], int(cls_pair[1])),
            csv.reader(f)))

    transforms = [
        A.Resize(*config['result_size'])]
    transforms = A.Compose(transforms)

    test_dir = Path(config['dataset']) / 'test'
    test_dset = PlantsDataset(
        test_dir, transforms=transforms, class_to_index=cls_to_id)
    num_classes = len(cls_to_id)

    # Get the model
    model = torchvision.models.resnext101_64x4d(
        weights=torchvision.models.ResNeXt101_64X4D_Weights.DEFAULT)
    layers = list(model.children())[:-1]
    layers.append(Flatten())
    layers.append(torch.nn.Linear(2048, out_features=num_classes))
    model = torch.nn.Sequential(*layers)
    model.to(device=device)

    model_params = torch.load(
        ckpt_dir / 'last_checkpoint.pth')['model_state_dict']
    model.load_state_dict(model_params)
    
    # Iterate over samples
    results = []
    with torch.no_grad():
        model.eval()
        try:
            for sample in tqdm(test_dset):
                image, target, _ = sample
                image = image[None, ...]
                image = image.to(device=device)
                predict = model(image)
                results.append(torch.argmax(predict.cpu()).item())
                # results.append(torch.argmax(predict.cpu()) == target)
        except KeyboardInterrupt:
            pass
        from collections import Counter
        print(Counter(results))
        # print(f'Total: {len(results)}')
        # print(f'Accuracy: {sum(results) / len(results)}')


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'config_pth', type=Path,
        help='Путь к конфигу обучения.')
    args = parser.parse_args(
        ['plants_classificator/configs/plant_classifier_1.json'])

    if not args.config_pth.exists():
        raise FileNotFoundError('Specified config file does not exists.')
    return args


if __name__ == "__main__":
    args = parse_args()
    config_pth = args.config_pth
    main(config_pth=config_pth)
