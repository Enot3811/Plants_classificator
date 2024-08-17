"""This module is designed to select gbif images from merged dataset.

All gbif images named by template "gbif_{genus}_{specie}_{n}.{img_ext}" so
they always can be selected and separated.
"""


from pathlib import Path
import sys
import shutil
import argparse

from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[3]))
from utils.data_utils.data_functions import collect_paths, IMAGE_EXTENSIONS


def select_gbif_images_from_merged_specie(
    specie_dir: Path, move_dir: Path, verbose: bool = False,
    ignore_existing: bool = False
):
    """Select gbif image from merged specie directory.

    Directory may contain images from several datasets
    and only gbif images will be selected and moved.

    Gbif images named as "gbif_{genus}_{specie}_{n}.{img_ext}".

    Parameters
    ----------
    specie_dir : Path
        Path to merged specie directory.
    move_dir : Path
        Path to move selected images.
    verbose : bool, optional
        Whether to show progress with tqdm. By default is `False`.
    ignore_existing : bool, optional
        Whether to ignore already existing directories and just delete them.
        By default is `False`.
    """
    # Check existing
    if move_dir.exists():
        if not ignore_existing:
            input(f'Specie dir "{str(move_dir)}" already exists. '
                  'If continue, this directory will be deleted. '
                  'Press enter to continue.')
        shutil.rmtree(move_dir)

    # Collect all the images
    img_pths = sorted(collect_paths(specie_dir, IMAGE_EXTENSIONS))

    # Iterate over paths
    if verbose:
        img_pths = tqdm(
            img_pths, desc=f'Selecting gbif from {str(specie_dir)}')
    for img_pth in img_pths:

        # Check every name by gbif pattern
        if len(img_pth.stem) > 4 and img_pth.stem[:4] == 'gbif':
            
            # Create dir only if there is some image to move
            if not move_dir.exists():
                move_dir.mkdir(parents=True)

            # And move
            dst_pth = move_dir / img_pth.name
            shutil.move(img_pth, dst_pth)


def select_gbif_images_from_merged_dataset(
    dataset_dir: Path, move_dir: Path, verbose: bool = False,
    ignore_existing: bool = False
):
    """Select gbif image from merged dataset.

    Dataset may contain images from several sources
    and only gbif images will be selected and moved.

    Gbif images named as "gbif_{genus}_{specie}_{n}.{img_ext}".

    Parameters
    ----------
    dataset_dir : Path
        Path to merged specie directory.
    move_dir : Path
        Path to move selected images.
    verbose : bool, optional
        Whether to show progress with tqdm. By default is `False`.
    ignore_existing : bool, optional
        Whether to ignore already existing directories and just delete them.
        By default is `False`.
    """
    # Prepare directory
    if move_dir.exists():
        if not ignore_existing:
            input(f'Specified directory "{str(move_dir)}" already exists. '
                  'If continue, this directory will be deleted. '
                  'Press enter to continue.')
        shutil.rmtree(move_dir)
    move_dir.mkdir(parents=True)

    # Iterate over species
    species_dirs = sorted(dataset_dir.iterdir())
    if verbose:
        species_dirs = tqdm(
            species_dirs, desc=f'Selecting gbif from {str(dataset_dir)}')
    for specie_dir in species_dirs:

        # Select gbif images from specie
        specie_move_dir = move_dir / specie_dir.name
        select_gbif_images_from_merged_specie(specie_dir, specie_move_dir)


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'dataset_dir', type=Path,
        help='Path to merged dataset dir.')
    parser.add_argument(
        'move_dir', type=Path,
        help='Path to move selected gbif images.')
    parser.add_argument(
        '--verbose', action='store_false',
        help='Whether to turn off verbose.')
    parser.add_argument(
        '--ignore_existing', action='store_true',
        help=('Whether to ignore already existing directories '
              'and just delete them.'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    select_gbif_images_from_merged_dataset(
        args.dataset_dir, args.move_dir, args.verbose, args.ignore_existing)
