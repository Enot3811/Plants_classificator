"""Download images using downloaded gbif meta data.

This script is a part of "gbif_download_images.py" script.
Check out its description to understand the general process.
"""


from pathlib import Path
import sys
import csv
import time
import requests
import multiprocessing as mp
from typing import List
import math
import argparse

from loguru import logger
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[3]))
from utils.argparse_utils import natural_int


def image_loader(
    multimedia_lines: List[List[str]], end_event: mp.Event,
    save_dir: Path, start_index: int
):
    """Handle function for parallel image loading.

    Parameters
    ----------
    multimedia_lines : List[List[str]]
        Lines from multimedia file in Darwin Core Archive format.
    end_event : mp.Event
        Event to stop loading.
    save_dir : Path
        Dir to save images.
    start_index : int
        Index of initial image for naming.
    """
    specie_name = save_dir.name.replace(' ', '_')
    try:
        i = start_index - 1  # -1 так как используется преинкремент
        for line in multimedia_lines:
            # Остановка скачивания, если семплов уже достаточно
            if end_event.is_set():
                break

            i += 1  # Индекс картинки
            format = line[2]  # Формат файла
            url = line[3]  # Ссылка
            # Если это изображение
            if (format == '' and url.split('.')[-1] == 'jpg' or
                    format == 'image/jpeg'):
                # Скачиваем
                img_pth = save_dir / f'gbif_{specie_name}_{i}.jpg'
                img_file = open(img_pth, 'wb')
                try:
                    # Делаем запрос
                    response = requests.get(url, stream=False, timeout=1)

                # Если timeout или ещё какая-то проблема
                except Exception as err:
                    logger.log('HTTP_IMG_ERR', f'{i} {str(err)}')
                    img_file.close()
                    img_pth.unlink(missing_ok=True)  # Удаляем неудачный файл
                    continue

                # Или получили плохой ответ
                if not response.ok:
                    logger.log('HTTP_IMG_ERR', f'{i} {str(response)}')
                    img_file.close()
                    img_pth.unlink(missing_ok=True)  # Удаляем неудачный файл
                    continue

                # Загружаем картинку поблочно
                for block in response.iter_content(None):
                    if not block:
                        break
                    img_file.write(block)
    except KeyboardInterrupt:
        if not img_file.closed:
            img_file.close()
            img_pth.unlink(missing_ok=True)  # Удаляем недозагруженный файл


def main(
    meta_dir: Path, num_workers: int, save_pth: Path,
    num_samples: int
):
    # Loguru
    logger.level("HTTP_IMG_ERR", no=35, color="<white>", icon="!")
    logger.remove()
    logger.add(sys.stdout, level=40, enqueue=True)
    logger.add(save_pth / 'log_file.log', level=0, enqueue=True)

    try:
        species = sorted(meta_dir.iterdir())
        for specie in species:
            logger.info(specie)
            save_dir = save_pth / 'images' / specie
            save_dir.mkdir(parents=True, exist_ok=True)

            # Читаем multimedia.txt и скачиваем картинки по url
            multimedia_file = meta_dir / specie / 'multimedia.txt'
            with open(multimedia_file) as f:
                iterator = iter(csv.reader(f, delimiter='\t'))
                next(iterator)  # Пропускаем шапку
                multimedia_lines = list(iterator)
            # multimedia_lines = [['', '', 'image/jpeg', 'https://purl.org/gbifnorway/img/ipt-specimens/barstow-garden/new/2013/P6281841.jpg'] for i in range(200)]  # noqa test
            length = len(multimedia_lines)
            lines_per_worker = math.ceil(length / num_workers)
            # Start workers
            end_event = mp.Event()
            workers: List[mp.Process] = []
            for i in range(num_workers):
                st = i * lines_per_worker
                ed = st + lines_per_worker
                worker = mp.Process(
                    target=image_loader, kwargs={
                        'multimedia_lines': multimedia_lines[st:ed],
                        'end_event': end_event,
                        'save_dir': save_dir,
                        'start_index': st
                    })
                worker.start()
                workers.append(worker)
            
            # Waiting for download
            progress_bar = tqdm(
                range(num_samples), f'Download "{specie}" images')
            while True:
                time.sleep(1)  # Check readiness every second
                current_imgs = len(list(save_dir.glob('*')))
                progress_bar.update(current_imgs - progress_bar.last_print_n)
                if (current_imgs > num_samples or
                        all(map(lambda proc: not proc.is_alive(), workers))):
                    end_event.set()
                    break
            progress_bar.close()
            logger.info(f'For {specie} downloaded '
                        f'{len(list(save_dir.glob("*")))} images')
            # Join workers
            for worker in workers:
                worker.join()

    except KeyboardInterrupt:
        print(f'Остановка на {specie=}')
        # Close remaining processes
        end_event.set()
    finally:
        logger.info('Производится остановка загрузчиков. '
                    'Это может занять какое-то время')
        for worker in workers:
            worker.join()
        progress_bar.close()


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'meta_dir', type=Path,
        help=('Path to directory that contain downloaded meta data '
              'for requested species.'))
    parser.add_argument(
        'save_pth', type=Path,
        help='Path to save downloaded data.')
    parser.add_argument(
        '--samples_per_cls', type=natural_int, default=300,
        help='Number of samples to download per each specie.')
    parser.add_argument(
        '--num_workers', type=natural_int, default=30,
        help='Number of parallel download processes.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(meta_dir=args.meta_dir, num_workers=args.num_workers,
         save_pth=args.save_pth, num_samples=args.samples_per_cls)
