from pathlib import Path


def main(dset_dir: Path):
    classes = [dir.name for dir in dset_dir.iterdir()]
    classes.sort()
    classes_txt = dset_dir / 'classes.txt'
    classes_txt.write_text('\n'.join(classes))


if __name__ == '__main__':
    dset_dir = Path('data/plants/inat2017_plants')
    main(dset_dir)
