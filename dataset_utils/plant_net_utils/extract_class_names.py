from pathlib import Path
import json


def main(dset_dir: Path):
    json_pth = dset_dir / 'plantnet300K_species_names.json'
    classes = list(set(json.loads(json_pth.read_text()).values()))
    classes.sort()
    classes = map(lambda x: x.replace('_', ' '), classes)
    classes_txt = dset_dir / 'classes.txt'
    classes_txt.write_text('\n'.join(classes))


if __name__ == '__main__':
    dset_dir = Path('data/plants/PlantNet300K')
    main(dset_dir)
