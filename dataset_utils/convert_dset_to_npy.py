"""Convert all images in PlantDataset to npy files."""

import sys
from pathlib import Path

from tqdm import tqdm
import numpy as np

sys.path.append(str(Path(__file__).parents[2]))
from plants_classificator.dataset_utils.plants_dataset import PlantsDataset


def main(dset_pth: Path, npy_dset_pth: Path):
    dset = PlantsDataset(dset_pth)
    id_to_cls = dset.get_index_to_class()
    for cls in tqdm(dset.get_class_to_index(), 'Create directories'):
        (npy_dset_pth / cls).mkdir(parents=True)
    for i in tqdm(range(len(dset)), 'Convert samples'):
        sample = dset.get_sample(i)
        image = sample['image']
        label = id_to_cls[sample['label']]
        img_name = sample['img_pth'].name.split('.')[0]
        dst_pth = npy_dset_pth / label / img_name
        np.save(dst_pth, image)


if __name__ == '__main__':
    dset_pth = Path('data/plants/inat17_clefeol17_plantnet300k')
    npy_dset_pth = Path('data/plants/inat17_clefeol17_plantnet300k_npy')
    main(dset_pth, npy_dset_pth)
