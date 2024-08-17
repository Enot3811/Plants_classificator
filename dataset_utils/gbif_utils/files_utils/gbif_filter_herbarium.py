"""Filter herbarium using trained YOLOv8-cls model.

In given gbif directory a new "herbarium_filter" subdir will be made.
It will contain "herbarium" and "living" subdirs
with filtered predicted images.
"""


from pathlib import Path
import shutil
import argparse
import sys

from ultralytics import YOLO
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[3]))
from utils.argparse_utils import natural_int


def main(model_pth: Path, gbif_dir: Path, batch_size: int):
    model = YOLO(model_pth)
    model.to(device='cuda')
    classes = ('herbarium', 'living')

    predict_dir = gbif_dir / 'herbarium_filter'

    specie_dirs = sorted((gbif_dir / 'images').iterdir())
    for specie_dir in tqdm(specie_dirs):
        save_dirs = [predict_dir / classes[0] / specie_dir.name,
                     predict_dir / classes[1] / specie_dir.name]
        save_dirs[0].mkdir(parents=True)
        save_dirs[1].mkdir(parents=True)

        img_pths = sorted(specie_dir.glob('*'))
        for i in range(0, len(img_pths), batch_size):
            samples = img_pths[i:i + batch_size]
            predicts = model(samples, verbose=False)
            for j, predict in enumerate(predicts):
                predicted_cls = predict[0].probs.top1
                dst_pth = save_dirs[predicted_cls] / img_pths[i + j].name
                shutil.move(img_pths[i + j], dst_pth)


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'model_pth', type=Path,
        help='Path to YOLO pt file.')
    parser.add_argument(
        'gbif_dir', type=Path,
        help='Path to gbif dir that contain "images" and "meta" subdirs.')
    parser.add_argument(
        '--b_size', type=natural_int, default=350,
        help='Batch size for the model.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(model_pth=args.model_pth, gbif_dir=args.gbif_dir,
         batch_size=args.b_size)
