"""Раскидать изображения по соответствующим их виду директориям.

Так как изображения названы по схеме "gbif_{genus}_{specie}_{n}.jpg",
можно легко определить какой директории принадлежит изображение.
"""


from pathlib import Path
import sys
import shutil

from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[3]))
from utils.data_utils.data_functions import collect_paths, IMAGE_EXTENSIONS


def main(images_dir: Path, gbif_images_dir: Path):

    img_pths = sorted(collect_paths(images_dir, IMAGE_EXTENSIONS))
    for img_pth in tqdm(img_pths):
        split_name = img_pth.stem.split('_')
        specie_name = split_name[1] + ' ' + split_name[2]

        dst_dir = gbif_images_dir / specie_name
        if not dst_dir.exists():
            input(f'"{str(dst_dir)}" did not found. Enter to continue')
            continue
        dst_pth = dst_dir / img_pth.name
        if dst_pth.exists():
            input(f'"{str(dst_pth)}" already exists. Enter to continue')
            continue
        shutil.move(img_pth, dst_pth)


if __name__ == '__main__':
    gbif_images_dir = Path('../data/plants/gbif/test/images')
    images_dir = Path('../data/plants/gbif/herbarium_detection/images')
    main(images_dir=images_dir, gbif_images_dir=gbif_images_dir)
