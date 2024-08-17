from pathlib import Path
import json
import shutil

from tqdm import tqdm


def main(dset_dir: Path):
    json_pth = dset_dir / 'plantnet300K_species_names.json'
    id_to_class = json.loads(json_pth.read_text())
    
    # Iterate over subsets
    subsets = ['images_test', 'images_val', 'images_train']
    for subset in subsets:
        subset_dir = dset_dir / subset
        # Iterate over all classes in subset
        for class_dir in tqdm(list(subset_dir.iterdir()), f'Process {subset}'):
            class_name = id_to_class[class_dir.name].replace('_', ' ')
            new_class_dir = class_dir.with_name(class_name)

            # Catch doubles
            if new_class_dir.exists():
                # Move each image separately
                for img_pth in class_dir.iterdir():
                    dst_pth = new_class_dir / img_pth.name
                    shutil.move(img_pth, dst_pth)
                shutil.rmtree(class_dir)

            else:
                shutil.move(class_dir, new_class_dir)


if __name__ == '__main__':
    dset_dir = Path('data/plants/PlantNet300k/')
    main(dset_dir)
