from pathlib import Path
import sys
import shutil

from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[3]))
from utils.data_utils.data_functions import IMAGE_EXTENSIONS, collect_paths


def main(dset_dir: Path):
    # Iterate over subsets
    # subsets = ['eol_data', 'web_train']
    subsets = ['eol_data']
    for subset in subsets:
        subset_dir = dset_dir / subset
        # Iterate over all classes in subset
        for class_dir in tqdm(list(subset_dir.iterdir()), f'Process {subset}'):
            sample_xml = next(iter(class_dir.glob('*.xml')))
            # 4 line, from 10 to -10 is class name
            cls_name = sample_xml.read_text().split('\n')[3][10:-10]
            cls_name = ' '.join(cls_name.split(' ')[:2])

            # Create new dir
            new_class_dir = class_dir.with_name(cls_name)
            new_class_dir.mkdir()

            # Move only images to new directory
            cls_images = collect_paths(class_dir, IMAGE_EXTENSIONS)
            for img_pth in cls_images:
                dst_pth = new_class_dir / img_pth.name
                shutil.move(img_pth, dst_pth)
            shutil.rmtree(class_dir)


if __name__ == '__main__':
    dset_dir = Path('data/plants/PlantCLEF_2017')
    main(dset_dir)
