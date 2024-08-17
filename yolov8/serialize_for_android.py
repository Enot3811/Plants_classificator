"""Serialize YOLOv8-cls model for android."""


import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.ao.quantization.quantize_jit import fuse_conv_bn_jit
from ultralytics import YOLO

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
from utils.data_utils.data_functions import read_image
from ultralytics.data.augment import CenterCrop, ToTensor
from torchvision.transforms import Compose, Normalize


def torch_serialize():
    model = YOLO('plants_classificator/yolov8/work_dir/inat17_inat21_clefeol17_plantnet300k_gbif_samples100_350_herb_25_100ep/weights/best.pt')
    model = model.model
    model.eval()
    example = torch.rand(1, 3, 224, 224)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module = fuse_conv_bn_jit(traced_script_module)
    # traced_script_module = torch.jit.script(model, example_inputs=example)
    traced_script_module = optimize_for_mobile(traced_script_module)
    save_name = 'traced.ptl'
    traced_script_module._save_for_lite_interpreter(save_name)
    traced_script_module = torch.jit.load(save_name)
    check_save(model, traced_script_module)


def check_save(model, traced_script_module,
               img_pth: str = '/home/pc0/Загрузки/Telegram Desktop/1.jpg'
):
    transforms = Compose([
        CenterCrop(224),
        ToTensor(),
        Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    ])
    samples = ['/home/pc0/Загрузки/Telegram Desktop/1.jpg']
    for img_pth in samples:
        img = read_image(img_pth, bgr_to_rgb=False)
        transformed_img = transforms(img)[None, ...]
        # transformed_img = transformed_img.to(device='cuda')
        model_out = torch.squeeze(model(transformed_img))
        traced_out = torch.squeeze(traced_script_module(transformed_img))
        model_idxs = torch.argsort(model_out, descending=True)
        traced_idxs = torch.argsort(traced_out, descending=True)
        print('Model', model_idxs[:5])
        print('Traced', traced_idxs[:5])
        # answer = [(id_to_cls[idx.item()], raw_out[idx].item())
        #         for idx in indexes[:5]]
        # for ans in answer:
        #     print(f'{ans[0]}: {ans[1]:.2f}')
        print()


def serialize_ultralytics():
    img_pth = '/home/pc0/Загрузки/Telegram Desktop/1.jpg'
    model = YOLO('plants_classificator/yolov8/work_dir/inat17_inat21_clefeol17_plantnet300k_gbif_samples100_350_herb_25_100ep/weights/best.pt')
    model.export(format="torchscript")  # creates 'yolov8n.torchscript'
    torchscript_model = YOLO("plants_classificator/yolov8/work_dir/inat17_inat21_clefeol17_plantnet300k_gbif_samples100_350_herb_25_100ep/weights/best.torchscript", task='classify')
    results1 = model(img_pth)
    img = read_image(img_pth, bgr_to_rgb=False)
    results2 = torchscript_model(img, imgsz=224)
    print()


if __name__ == '__main__':
    torch_serialize()
    # serialize_ultralytics()
