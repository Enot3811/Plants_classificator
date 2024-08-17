import sys
from pathlib import Path
from typing import Dict, Tuple, Sequence, Any, List, Optional

import torch
import numpy as np
import torch.utils
import torch.utils.data

sys.path.append(str(Path(__file__).parents[2]))
from utils.torch_utils.datasets import AbstractClassificationDataset
from utils.torch_utils.torch_functions import image_numpy_to_tensor
from utils.data_utils.data_functions import read_image


class PlantsDataset(AbstractClassificationDataset):

    def _parse_dataset_pth(self, dset_pth: Path | str) -> Path:
        return super()._parse_dataset_pth(dset_pth)

    def _collect_samples(self, dset_pth: Path) -> Sequence[Path]:
        samples = []
        for img_ext in ['jpg', 'jpeg', 'png', 'npy', 'JPG', 'JPEG', 'PNG']:
            samples += list(dset_pth.glob('*/*.' + img_ext))
        return samples
        
    def _collect_class_labels(
        self,
        samples: Sequence[Path],
        class_to_index: Optional[Dict[str, int]] = None
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Labels are directories
        labels = set(map(lambda dir_pth: dir_pth.name,
                         self.dset_pth.glob('*')))
        if class_to_index is not None:
            if not labels.issubset(class_to_index.keys()):
                raise ValueError(
                    'Passed "class_to_index" does not contain all classes '
                    'represented in the dataset.')
        else:
            class_to_index = {label: i for i, label in enumerate(labels)}
        index_to_class = {
            idx: label for label, idx in class_to_index.items()}
        return class_to_index, index_to_class

    def get_sample(self, index: int) -> Dict[str, Any]:
        """Read image, convert class name to id and pack it to dict.

        Parameters
        ----------
        index : int
            Index of sample.

        Returns
        -------
        Dict[str, Any]
            Dict with "image", "label" and "img_pth".
        """
        img_pth = super().get_sample(index)
        if img_pth.name.split('.')[-1] == 'npy':
            image = np.load(img_pth)
        else:
            image = read_image(img_pth)
        label = self._class_to_index[img_pth.parent.name]
        return {'image': image, 'label': label, 'img_pth': img_pth}
    
    def apply_transforms(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply passed transforms on the sample.

        Parameters
        ----------
        sample : Dict[str, Any]
            Sample to transform.

        Returns
        -------
        Dict[str, Any]
            Transformed sample.
        """
        if self.transforms:
            augmented = self.transforms(
                image=sample['image'], label=sample['label'])
            sample['image'] = augmented['image']
            sample['label'] = augmented['label']
        return sample
    
    def postprocess_sample(
        self, sample: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, Path]:
        """Make postprocessing for sample after getting and augmentations.
        
        Convert image and label to tensors.

        Parameters
        ----------
        sample : Dict[str, Any]
            Dict with "image", "label" and "img_pth".

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, Path]
            Sample in torch compatible view.
        """
        image = image_numpy_to_tensor(sample['image']) / 255
        label = torch.tensor(sample['label'])
        return image, label, sample['img_pth']
        
    @staticmethod
    def collate_func(batch: List[Any]) -> Any:
        images, labels, pths = tuple(map(list, zip(*batch)))
        images = torch.stack(images)
        labels = torch.stack(labels)
        return (images, labels, pths)


if __name__ == '__main__':
    # Test script
    import albumentations as A
    from utils.data_utils.data_functions import show_images_cv2
    from utils.torch_utils.torch_functions import image_tensor_to_numpy

    # Parameters
    dset_pth = 'data/plants/test/train'
    test_iter = 5
    horizontal_flip = True
    vertical_flip = True
    blur = False
    color_jitter = True
    img_size = (224, 224)
    b_size = 4
    show_images = True
    source_cycle = False
    torch_cycle = False
    dloader_cycle = True
    shuffle_dloader = True

    # Get augmentations
    transforms = [A.RandomResizedCrop(*img_size, scale=(0.8, 1.0))]
    if horizontal_flip:
        transforms.append(A.HorizontalFlip())
    if vertical_flip:
        transforms.append(A.VerticalFlip())
    if blur:
        transforms.append(A.Blur(blur_limit=3, p=0.5))
    if color_jitter:
        transforms.append(A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0, p=0.2))
    transforms = A.Compose(transforms)

    # Get dataset
    dset = PlantsDataset(dset_pth, transforms=transforms)
    id_to_cls = dset.get_index_to_class()

    # Numpy iterating
    if source_cycle:
        for i in range(len(dset)):
            sample = dset.get_sample(i)
            image = sample['image']
            label = id_to_cls[sample['label']]
            img_pth = sample['img_pth']
            print(image.shape, label, str(img_pth))
            if show_images:
                show_images_cv2(image, destroy_windows=False, delay=2000)
            if i == test_iter:
                break
    # Torch iterating
    if torch_cycle:
        for i, sample in enumerate(dset):
            image, label, img_pth = sample
            print(image.shape, label, str(img_pth))
            if show_images:
                image = image_tensor_to_numpy(image)
                show_images_cv2(image, destroy_windows=False, delay=2000)
            if i == test_iter:
                break
    # DLoader cycle
    if dloader_cycle:
        dloader = torch.utils.data.DataLoader(
            dset, b_size, shuffle_dloader, collate_fn=dset.collate_func)
        for i, batch in enumerate(dloader):
            images, labels, pths = batch
            print(images.shape, labels)
            if show_images:
                images = [image_tensor_to_numpy(images[j])
                          for j in range(len(images))]
                show_images_cv2(images)
            if i == test_iter:
                break
