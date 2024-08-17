"""Iterate over dataset to check data integrity."""

import sys
from pathlib import Path

from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[1]))
from dataset_utils.plants_dataset import PlantsDataset


def main(dset_pth: Path):
    dset = PlantsDataset(dset_pth)
    for i in tqdm(range(len(dset)), 'Iterate dataset'):
        dset.get_sample(i)


if __name__ == '__main__':
    dset_pth = Path('data/plants/inat21/train')
    main(dset_pth)
