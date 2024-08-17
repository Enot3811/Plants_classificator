"""Resize downloaded images.

Image file will be deleted if it has wrong format or corrupted.
"""


from pathlib import Path
import sys
import argparse

import albumentations as A
from loguru import logger
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[3]))
from utils.data_utils.data_functions import (
    read_image, save_image, collect_paths, IMAGE_EXTENSIONS)


def main(species_dir: Path):
    logger.remove()
    logger.add(sys.stdout, level=0, enqueue=True)
    logger.add('resize_log.log', level=0, enqueue=True)

    resizer = A.SmallestMaxSize(600)
    try:
        species_dirs = sorted(species_dir.iterdir())
        for images_dir in species_dirs:
            img_pths = sorted(collect_paths(images_dir, IMAGE_EXTENSIONS))
            for img_pth in tqdm(img_pths):
                try:
                    img = read_image(img_pth)
                    shape = img.shape
                    if len(shape) != 3:
                        logger.warning(f'{str(img_pth)} wrong shape')
                        continue
                except Exception as err:
                    logger.error(f'{str(img_pth)} {err}')
                    img_pth.unlink(missing_ok=True)
                    continue
                resized = resizer(image=img)['image']
                save_image(resized, img_pth)
            res_num = len(list(images_dir.glob('*')))  # Check remaining images
            src_num = len(img_pths)
            logger.info(f'{images_dir.name} {src_num=} {res_num=}')
            if src_num - res_num > 70:
                logger.warning(
                    f'{images_dir.name} deleted more than 70 samples')
                    
    except KeyboardInterrupt:
        logger.info(f'Interrupt on {str(img_pth)}')


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'species_dir', type=Path,
        help=('Directory with species subdirectories '
              'containing images to resize.'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(species_dir=args.species_dir)
