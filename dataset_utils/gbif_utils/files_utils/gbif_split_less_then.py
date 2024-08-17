"""Отобрать gbif виды по количеству семплов.

Если меньше указанного порога, то они будут перемещены в новую директорию.
"""


import sys
from pathlib import Path
import shutil

from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[3]))
from utils.data_utils.data_functions import collect_paths, IMAGE_EXTENSIONS


def main(gbif_dir: Path, move_dir: Path, threshold: int, move_meta: bool):
    gbif_images_dir = gbif_dir / 'images'
    gbif_meta_dir = gbif_dir / 'meta'

    moved_imgs_dir = move_dir / 'images'
    moved_imgs_dir.mkdir(parents=True)
    if move_meta:
        moved_meta_dir = move_dir / 'meta'
        moved_meta_dir.mkdir(parents=True)

    species = sorted(map(
        lambda x: x.stem,
        filter(lambda x: x.is_dir(), gbif_images_dir.glob('*'))))
    for specie in tqdm(species):
        specie_img_dir = gbif_images_dir / specie
        if move_meta:
            specie_meta_dir = gbif_meta_dir / specie

        specie_imgs = collect_paths(specie_img_dir, IMAGE_EXTENSIONS)
        if len(specie_imgs) < threshold:
            shutil.move(specie_img_dir, moved_imgs_dir / specie_img_dir.name)
            if move_meta:
                shutil.move(specie_meta_dir,
                            moved_meta_dir / specie_meta_dir.name)


if __name__ == '__main__':
    gbif_dir = Path('../data/plants/inat17_inat21_clefeol17_plantnet300k_gbif')
    move_dir = Path('../data/plants/inat17_inat21_clefeol17_plantnet300k_gbif/'
                    'inat17_inat21_clefeol17_plantnet300k_gbif_less100')
    threshold = 100
    move_meta = False
    main(gbif_dir=gbif_dir, move_dir=move_dir, threshold=threshold,
         move_meta=move_meta)
