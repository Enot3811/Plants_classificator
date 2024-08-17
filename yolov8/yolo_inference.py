"""Infer YOLOv8 for plants."""


from pathlib import Path
import sys
from typing import Union

from ultralytics import YOLO

sys.path.append(str(Path(__file__).parents[2]))
from utils.data_utils.data_functions import (
    show_images_cv2, prepare_path, resize_image, collect_paths,
    IMAGE_EXTENSIONS)


def infer_directory(image_dir: Union[Path, str], model_pth: Union[Path, str]):
    """Iterate over images in directory and make a predict with model.

    Parameters
    ----------
    image_dir : Union[Path, str]
        Path to images.
    model_pth : Union[Path, str]
        Path to YOLO checkpoint.
    """
    image_dir = prepare_path(image_dir)
    model = YOLO(model_pth)
    for img_pth in sorted(collect_paths(image_dir, IMAGE_EXTENSIONS)):
        result = model(img_pth, verbose=True, show=False, save=False)[0]
        max_side = max(result.orig_shape)
        if max_side > 1000:
            resize_ratio = 1000 / max_side
            new_size = tuple(
                map(lambda x: int(x * resize_ratio), result.orig_shape))
            img = resize_image(result.orig_img, new_size)
        else:
            img = result.orig_img
        key = show_images_cv2(
            img, 'image_to_predict', destroy_windows=False, rgb_to_bgr=False)
        if key == 27:
            break


if __name__ == '__main__':
    # Configure parameters and watch results
    image_dir = '../data/plants/camera/'
    model_pth = (
        'plants_classificator/yolov8/work_dir/'
        'inat17_inat21_clefeol17_plantnet300k_gbif_samples100_350_herb_25_100ep/'
        'weights/best.pt')
    infer_directory(image_dir, model_pth)
