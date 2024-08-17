from pathlib import Path


def main(dset_dir: Path):
    classes = []
    for dir in dset_dir.iterdir():
        sample_xml = next(iter(dir.glob('*.xml')))
        # 4 line, from 10 to -10 is class name
        cls_name = sample_xml.read_text().split('\n')[3][10:-10]
        cls_name = ' '.join(cls_name.split(' ')[:2])
        classes.append(cls_name)
    classes.sort()
    classes_txt = dset_dir.with_name('classes.txt')
    classes_txt.write_text('\n'.join(classes))


if __name__ == '__main__':
    dset_dir = Path('data/plants/PlantCLEF_2017/eol_data')
    main(dset_dir)
