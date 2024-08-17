"""Добавить к названиям картинок gbif и название вида.

123.jpg -> gbif_genus_specie_123.jpg
Некоторые изображения назывались как "{номер таксона}_{номер}.jpg
321_123.jpg -> gbif_genus_specie_321_123.jpg
"""


from pathlib import Path

from tqdm import tqdm


def main(gbif_dir: Path):
    gbif_images = gbif_dir / 'images'
    species_dirs = sorted(gbif_images.glob('*'))
    for specie_dir in tqdm(species_dirs):
        for img_pth in sorted(specie_dir.iterdir()):
            if 'gbif' not in img_pth.name:
                specie = '_'.join(specie_dir.name.split(' '))
                new_name = f'gbif_{specie}_{img_pth.name}'
                img_pth.rename(img_pth.with_name(new_name))


if __name__ == '__main__':
    gbif_dir = Path('../data/plants/gbif/gbif')
    main(gbif_dir=gbif_dir)
