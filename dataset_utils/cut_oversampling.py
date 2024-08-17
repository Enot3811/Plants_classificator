"""Убрать лишние семплы из видов, для которых их больше порогового значения."""

from pathlib import Path
import sys
import shutil

from tqdm import tqdm
import random

sys.path.append(str(Path(__file__).parents[2]))
from utils.data_utils.data_functions import collect_paths, IMAGE_EXTENSIONS


def main(species_dir: Path, move_dir: Path, threshold: int, random_seed: int):
    random.seed = random_seed

    for specie_dir in tqdm(sorted(species_dir.iterdir(),
                                  key=lambda x: x.name)):
        img_pths = collect_paths(specie_dir, IMAGE_EXTENSIONS)
        if len(img_pths) > threshold:
            pths_to_move = (
                set(img_pths).difference(random.sample(img_pths, threshold)))
            dst_dir = move_dir / specie_dir.name
            dst_dir.mkdir(parents=True)
            for pth in pths_to_move:
                shutil.move(pth, dst_dir)


if __name__ == '__main__':
    move_dir = Path('../data/plants/inat17_inat21_clefeol17_plantnet300k_gbif/'
                    'oversampling_more_than_350/images')
    species_dir = Path(
        '../data/plants/inat17_inat21_clefeol17_plantnet300k_gbif/images')
    threshold = 350
    random_seed = 42
    main(species_dir=species_dir, move_dir=move_dir, threshold=threshold,
         random_seed=random_seed)
