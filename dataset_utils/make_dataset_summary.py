"""Make dataset summary.

Csv file contains id, samples counts and weights per classes.
Txt contains min, max, mean and median sample counts.
"""

import sys
from pathlib import Path
import csv

from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

sys.path.append(str(Path(__file__).parents[2]))
from utils.data_utils.data_functions import collect_paths, IMAGE_EXTENSIONS


def main(dset_pth: Path):
    categories_dirs = dset_pth.glob('*')
    counter = {}
    for category_dir in tqdm(categories_dirs):
        counter[category_dir.name] = len(
            collect_paths(category_dir, IMAGE_EXTENSIONS))
    sorted_counter = sorted(list(counter.items()), key=lambda pair: pair[1])
    dset_len = sum(counter.values())
    median_cnt = sorted_counter[len(sorted_counter) // 2][-1]
    min_cnt = sorted_counter[0][-1]
    max_cnt = sorted_counter[-1][-1]
    mean_cnt = dset_len / len(counter)
    dset_labels = []
    # Make y for compute_class_weight
    for i, (species, count) in enumerate(sorted_counter):
        dset_labels += [i] * count
    weights = compute_class_weight(
        'balanced', classes=np.array(range(len(sorted_counter))),
        y=dset_labels)
    # Save weights to counter
    for i, (species, count) in enumerate(sorted_counter):
        sorted_counter[i] = (species, count, weights[i])
    # Save csv
    csv_save_pth = dset_pth.with_name(dset_pth.name + '.csv')
    if csv_save_pth.exists():
        input(f'CSV file {str(csv_save_pth)} already exists.'
              'If continue it will be deleted.')
    with open(csv_save_pth, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(sorted_counter)
    # Save some stats
    stats = [
        f'Min count: {min_cnt}\n',
        f'Max count: {max_cnt}\n',
        f'Mean count: {mean_cnt}\n',
        f'Median count: {median_cnt}\n']
    txt_save_pth = dset_pth.with_name(dset_pth.name + '.txt')
    if txt_save_pth.exists():
        input(f'TXT file {str(txt_save_pth)} already exists.'
              'If continue it will be deleted.')
    with open(txt_save_pth, 'w') as f:
        f.writelines(stats)


if __name__ == '__main__':
    dset_pth = Path(
        '../data/plants/inat17_inat21_clefeol17_plantnet300k_gbif/images')
    main(dset_pth)
