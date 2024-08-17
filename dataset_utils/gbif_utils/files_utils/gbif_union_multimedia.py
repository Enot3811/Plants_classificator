"""Склеивание multimedia файлов из старого формата скачивания в один."""


from typing import List
from pathlib import Path
import csv

from tqdm import tqdm


def main(meta_dirs: List[Path]):
    for meta_dir in meta_dirs:
        for specie_dir in tqdm(list(meta_dir.iterdir()),
                               f'Process {str(meta_dir)}'):
            csv_lines = None
            for taxon_dir in sorted(filter(lambda x: x.is_dir(),
                                           specie_dir.glob('*')),
                                    key=lambda x: int(x.name)):
                with open(taxon_dir / 'multimedia.txt') as f:
                    if csv_lines is None:
                        csv_lines = list(csv.reader(f, delimiter='\t'))
                    else:
                        csv_lines += list(csv.reader(f, delimiter='\t'))[1:]
            with open(specie_dir / 'multimedia.txt', 'w') as f:
                csv.writer(f, delimiter='\t').writerows(csv_lines)


if __name__ == '__main__':
    meta_dirs = [
        Path('../data/plants/gbif/old_naming/meta')
    ]
    main(meta_dirs=meta_dirs)
