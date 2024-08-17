from pathlib import Path
import shutil

from tqdm import tqdm


def normalize_dataset(dataset_dir: Path):
    # Iterate over subsets
    subsets = ['train', 'val']
    for subset in subsets:
        subset_dir = dset_dir / subset
        # Iterate over all classes in subset
        for class_dir in tqdm(list(subset_dir.iterdir()), f'Process {subset}'):
            class_name = ' '.join(class_dir.name.split('_')[-2::1])
            new_class_dir = class_dir.with_name(class_name)

            # Catch doubles
            if new_class_dir.exists():
                raise
            else:
                shutil.move(class_dir, new_class_dir)


def main(dset_dir: Path):
    # Iterate over subsets
    subsets = ['train', 'val']
    for subset in subsets:
        subset_dir = dset_dir / subset
        # Iterate over all classes in subset
        for class_dir in tqdm(list(subset_dir.iterdir()), f'Process {subset}'):
            class_name = ' '.join(class_dir.name.split('_')[-2::1])
            new_class_dir = class_dir.with_name(class_name)

            # Catch doubles
            if new_class_dir.exists():
                raise
            else:
                shutil.move(class_dir, new_class_dir)


if __name__ == '__main__':
    dset_dir = Path('data/plants/inat21')
    main(dset_dir)
