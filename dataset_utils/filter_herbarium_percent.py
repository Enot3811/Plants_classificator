"""Отобрать виды, в которых гербариума меньше заданного процента."""

from pathlib import Path
import shutil
import csv

from ultralytics import YOLO
from tqdm import tqdm


def main(model_pth: Path, species_dir: Path, batch_size: int,
         threshold_percent: float, filter_dir: Path):
    model = YOLO(model_pth)
    model.to(device='cuda')
    required_living_percent = 1.0 - threshold_percent
    filter_dir.mkdir(parents=True)
    predicts_dir = species_dir.parent / 'predicts'
    predicts_dir.mkdir()
    classes = ['herbarium', 'living']

    for specie_dir in tqdm(sorted(species_dir.iterdir(),
                                  key=lambda x: x.name)):
        img_pths = sorted(specie_dir.glob('*'))
        f = open(predicts_dir / (specie_dir.name + '.csv'), 'w')
        writer = csv.writer(f)  # Записываем путь и предсказанный класс
        pred_idxs = []  # Собираем предсказанные индексы
        for i in range(0, len(img_pths), batch_size):
            samples = img_pths[i:i + batch_size]
            predicts = model(samples, verbose=False)
            for pred in predicts:
                pred_idxs.append(pred.probs.top1)
                writer.writerow(
                    [Path(pred.path).name, classes[pred.probs.top1]])
        f.close()
        living_percent = sum(pred_idxs) / len(predicts)
        if living_percent < required_living_percent:
            shutil.move(specie_dir, filter_dir)


if __name__ == '__main__':
    model_pth = Path(
        'plants_classificator/yolov8/work_dir/herbarium_100ep/weights/best.pt')
    species_dir = Path(
        '../data/plants/inat17_inat21_clefeol17_plantnet300k_gbif/images')
    batch_size = 400
    threshold_percent = 0.25
    filter_dir = Path(
        '../data/plants/inat17_inat21_clefeol17_plantnet300k_gbif/'
        'herbarium_percent_more25/images')
    main(model_pth=model_pth, species_dir=species_dir, batch_size=batch_size,
         threshold_percent=threshold_percent, filter_dir=filter_dir)
