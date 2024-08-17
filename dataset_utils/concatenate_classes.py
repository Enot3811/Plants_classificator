"""Make a list of classes from several datasets."""

from typing import List
from pathlib import Path


def main(classes_txts: List[Path]):
    
    classes = [set(file.read_text().split('\n')) for file in classes_txts]
    lens = list(map(len, classes))
    print(lens)

    inat_plantnet = classes[0].union(classes[1])
    whole = list(inat_plantnet.union(classes[2]))
    whole.sort()
    print(len(whole))
    classes_txt = classes_txts[0].parents[1] / 'classes.txt'
    classes_txt.write_text('\n'.join(whole))


if __name__ == '__main__':
    classes_txts = [
        Path('data/plants/inat2017_plants/classes.txt'),
        Path('data/plants/plantnet_300K/classes.txt'),
        Path('data/plants/PlantCLEF_2017/classes.txt')]
    main(classes_txts)
