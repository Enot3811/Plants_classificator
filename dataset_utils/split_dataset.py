"""Split dataset according to proportions.

The division is carried out in such a way that samples of each class
are divided according to proportions.
"""


import sys
from pathlib import Path
import shutil
import argparse

from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[1]))
from utils.data_utils.data_functions import IMAGE_EXTENSIONS, collect_paths
from utils.argparse_utils import unit_interval


def main(dset_pth: Path, dst_pth: Path, proportions):

    # Check and create directories
    if dst_pth.exists():
        input(f'Specified directory "{str(dst_pth)}" already exists. '
              'Сontinuing to work will delete the data located there. '
              'Press enter to continue.')
        shutil.rmtree(dst_pth)
    dst_pth.mkdir()
    splits_dirs = []
    for i in range(len(proportions)):
        split_dir = dst_pth / f'split{i}'
        splits_dirs.append(split_dir)
        split_dir.mkdir()

    # Iterate over classes in source dataset
    for cls_dir in tqdm(list(dset_pth.iterdir()), 'Split dataset'):
        cls_samples = collect_paths(cls_dir, IMAGE_EXTENSIONS)

        remain = len(cls_samples)
        for i, proportion in enumerate(proportions):
            # Calculate number of samples per split
            split_len = int(len(cls_samples) * proportion)
            if split_len == 0 and remain != 0:
                split_len = 1
            remain -= split_len
        
            # Copy files to split dir
            split_cls_dir = splits_dirs[i] / cls_dir.name
            split_cls_dir.mkdir()
            for cls_sample in cls_samples[:split_len]:
                shutil.move(cls_sample, split_cls_dir)
                # shutil.copy2(cls_sample, split_cls_dir)
            # Cut out copied files
            cls_samples = cls_samples[split_len:]
        # Check remained samples and put them to the last split
        else:
            if len(cls_samples) != 0:
                for cls_sample in cls_samples:
                    shutil.move(cls_sample, split_cls_dir)
                    # shutil.copy2(cls_sample, split_cls_dir)


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'dset_pth', type=Path,
        help='Directory with species subdirectories containing images.')
    parser.add_argument(
        'dst_pth', type=Path,
        help='Path to move split dataset.')
    parser.add_argument(
        'proportions', type=unit_interval, nargs='+',
        help='Proportions for split.')
    args = parser.parse_args()

    if not args.dset_pth.exists():
        raise FileNotFoundError(
            f'Specified directory "{str(args.dset_pth)}" does not exist.')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(dset_pth=args.dset_pth, dst_pth=args.dst_pth,
         proportions=args.proportions)
