"""Привести старые имена изображений к новому формату.

Старый формат: "{номер таксона}_{номер}.jpg" и несколько multimedia.txt
Ранее все multimedia уже сливались в единый файл, поэтому новый формат
"{номер}.jpg" будет вести нумерацию по нему.
"""


from pathlib import Path

from tqdm import tqdm


def main(gbif_dir: Path):
    gbif_images = gbif_dir / 'images'
    gbif_metas = gbif_dir / 'meta'
    species = sorted(map(lambda x: x.name, gbif_metas.iterdir()))
    for specie in species:
        specie_imgs = gbif_images / specie
        specie_meta = gbif_metas / specie

        taxons_metas = sorted(
            filter(lambda x: x.is_dir(), specie_meta.glob('*')),
            key=lambda x: int(x.name)
        )
        taxon_st_idx = 0
        for taxon_meta in taxons_metas:
            taxon_id = taxon_meta.name
            with open(taxon_meta / 'multimedia.txt') as f:
                taxon_len = len(f.readlines()) - 1
            taxon_ed_idx = taxon_st_idx + taxon_len

            taxon_images = specie_imgs.glob(f'{taxon_id}*')
            for taxon_image in tqdm(taxon_images, desc=specie):
                img_n = int(taxon_image.stem.split('_')[-1])
                abs_img_n = taxon_st_idx + img_n
                new_name = str(abs_img_n) + taxon_image.suffix
                taxon_image.rename(taxon_image.with_name(new_name))

            taxon_st_idx = taxon_ed_idx


if __name__ == '__main__':
    gbif_dir = Path('../data/plants/gbif/old_naming')
    main(gbif_dir=gbif_dir)
