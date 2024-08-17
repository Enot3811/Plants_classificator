"""Download images of specified species from gbif.

Downloading from gbif can be described in several steps:
1) Making a request to gbif with pygbif and multiprocessing libs.
One account in gbif can have only 3 parallel requests.
Gbif responds to a request on average in 1-2 minutes, but sometimes a request
can hang for a long time.
2) Download requested meta data from gbif.
Meta data organized in "Darwin Core Archive" format.
"multimedia.txt" is a main file for getting images. It contain lines with
specie occurrences that may have link to an image.
3) Iterate over multimedia file and download several images simultaneously with
multiprocessing to overcome low downloading speed.
"""


import time
import csv
import requests
from pathlib import Path
import zipfile
import shutil
from typing import List
import multiprocessing as mp
import math
import sys
import argparse

import pygbif.occurrences as occ
import pygbif.species as spc
from tqdm import tqdm
from loguru import logger

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


def meta_loader(
    species: List[str], meta_queue: mp.Queue,
    save_dir: Path, num_gbif_requests: int,
    gbif_name: str, gbif_pwd: str, gbif_mail: str
):
    """Handle function for gbif requests.

    Parameters
    ----------
    species : List[str]
        List with species to make gbif requests.
    meta_queue : mp.Queue
        Queue that used for sending ready gbif meta directories.
    save_dir : Path
        Directory to save downloaded meta.
    num_gbif_requests : int
        Number of parallel gbif requests per time.
    """
    try:
        requests_ids = []
        while len(species) != 0 or len(requests_ids) != 0:

            # Make parallel gbif requests
            while len(requests_ids) < num_gbif_requests and len(species) != 0:
                specie = species.pop(0)

                # Берём id таксонов и составляем запрос на выкачку
                taxons_meta = None
                while taxons_meta is None:
                    try:
                        taxons_meta = spc.name_suggest(specie)
                    except requests.HTTPError as err:
                        logger.log('HTTP_GBIF_ERR', f'{specie=} {err}')
                        time.sleep(60)
                    except Exception as err:
                        logger.log(
                            'UNRECOGNIZED_GBIF_ERR', f'{specie=} {str(err)}')
                        time.sleep(60)

                # If some specie was not represented in gbif
                if len(taxons_meta) == 0:
                    logger.log('GBIF_SPECIE_NOT_FOUND',
                               f'{specie=} is not found.')
                    continue

                taxons_ids = map(lambda x: str(x['key']), taxons_meta)
                taxons_ids = map(lambda x: f'"{x}"', taxons_ids)
                taxons_ids = ', '.join(taxons_ids)

                result = None
                while result is None:
                    try:
                        result = occ.download(
                            [f'taxonKey in [{taxons_ids}]'], format='DWCA',
                            user=gbif_name, pwd=gbif_pwd,
                            email=gbif_mail)
                    except requests.HTTPError as err:
                        logger.log('HTTP_GBIF_ERR', f'{specie=} {err}')
                        time.sleep(60)
                    except Exception as err:
                        logger.log(
                            'UNRECOGNIZED_GBIF_ERR', f'{specie=} {str(err)}')
                download_id = result[0]
                requests_ids.append((specie, download_id))

            # Check requests' readiness while there is no any ready one
            ready_id = None
            while ready_id is None:
                for i, (specie, request_id) in enumerate(requests_ids):
                    try:
                        meta = occ.download_meta(request_id)
                    except requests.HTTPError as err:
                        logger.log('HTTP_GBIF_ERR', f'{specie=} {str(err)}')
                        continue
                    except Exception as err:
                        logger.log(
                            'UNRECOGNIZED_GBIF_ERR', f'{specie=} {str(err)}')
                    if meta["status"] == "SUCCEEDED":
                        ready_id = request_id
                        requests_ids.pop(i)
                        break
                    else:
                        msg = f'{specie=} Попытка скачать'
                        logger.info(msg)
                else:
                    time.sleep(60)

            # Download and unpack zip
            specie_dir = save_dir / specie
            specie_dir.mkdir(exist_ok=True, parents=True)
            zip_pth = specie_dir / f'{ready_id}.zip'
            while not zip_pth.exists():
                try:
                    occ.download_get(ready_id, specie_dir)
                except Exception as err:
                    logger.log(
                        'UNRECOGNIZED_GBIF_ERR', f'{specie=} {str(err)}')
                    time.sleep(60)
            with zipfile.ZipFile(zip_pth, 'r') as f:
                f.extractall(specie_dir)

            # Wait until the loaders process the old data
            while meta_queue.full():
                time.sleep(5)
            # Send metadata dir
            meta_queue.put(specie_dir)

    except KeyboardInterrupt:
        pass


def main(species_file: Path, save_pth: Path, samples_per_cls: int,
         num_workers: int, num_gbif_requests: int, gbif_name: str,
         gbif_pwd: str, gbif_mail: str):
    # Prepare paths, dirs and logs
    meta_pth = save_pth / 'meta'
    images_dir = save_pth / 'images'
    if save_pth.exists():
        input(f'"{str(save_pth)}" will be deleted.')
        shutil.rmtree(save_pth)
    save_pth.mkdir()
    meta_pth.mkdir()
    images_dir.mkdir()

    # Loguru
    logger.level("HTTP_IMG_ERR", no=35, color="<white>", icon="!")
    logger.level("HTTP_GBIF_ERR", no=40, color="<yellow>", icon="!!")
    logger.level("UNRECOGNIZED_GBIF_ERR", no=45, color="<red>", icon="!!!")
    logger.level("GBIF_SPECIE_NOT_FOUND", no=45, color="<blue>", icon="?")
    logger.remove()
    logger.add(sys.stdout, level=40, enqueue=True)
    logger.add(save_pth / 'log_file.log', level=0, enqueue=True)

    try:
        # Prepare species list and start gbif loader
        with open(species_file) as f:
            species = f.readlines()
        species = list(map(lambda specie: specie.strip(), species))
        meta_queue = mp.Queue(num_gbif_requests)
        gbif_worker = mp.Process(target=meta_loader, kwargs={
            'species': species,
            'meta_queue': meta_queue,
            'save_dir': meta_pth,
            'num_gbif_requests': num_gbif_requests,
            'gbif_name': gbif_name,
            'gbif_pwd': gbif_pwd,
            'gbif_mail': gbif_mail
        })
        gbif_worker.start()

        # Work while there is species in gbif loader (or some remain in queue)
        while gbif_worker.is_alive() or not meta_queue.empty():
            # Get meta data dir
            meta_dir = meta_queue.get()
            specie = meta_dir.name

            logger.info(specie)
            specie_imgs = images_dir / specie
            specie_imgs.mkdir(exist_ok=True)

            # Читаем multimedia.txt и скачиваем картинки по url
            with open(meta_dir / 'multimedia.txt') as f:
                iterator = iter(csv.reader(f, delimiter='\t'))
                next(iterator)  # Пропускаем шапку
                multimedia_lines = list(iterator)
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
                        'save_dir': specie_imgs,
                        'start_index': st
                    })
                worker.start()
                workers.append(worker)
            
            # Waiting for download
            progress_bar = tqdm(
                range(samples_per_cls), f'Download "{specie}" images')
            while True:
                time.sleep(1)  # Check readiness every second
                current_imgs = len(list(specie_imgs.glob('*')))
                progress_bar.update(current_imgs - progress_bar.last_print_n)
                if (current_imgs > samples_per_cls or
                        all(map(lambda proc: not proc.is_alive(), workers))):
                    end_event.set()
                    break
            progress_bar.close()
            logger.info(f'For {specie} downloaded '
                        f'{len(list(specie_imgs.glob("*")))} images')
            # Join workers
            for worker in workers:
                worker.join()

    except KeyboardInterrupt:
        logger.info(f'Остановка на {specie=}')
        # Close remaining processes
        end_event.set()
    finally:
        logger.info('Производится остановка загрузчиков. '
                    'Это может занять какое-то время')
        for worker in workers:
            worker.join()
        gbif_worker.join()
        progress_bar.close()


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'species_file', type=Path,
        help='CSV file with species to download from gbif.')
    parser.add_argument(
        'save_pth', type=Path,
        help='Path to save downloaded data.')
    parser.add_argument(
        'gbif_name', type=str,
        help='Gbif user name.')
    parser.add_argument(
        'gbif_pwd', type=str,
        help='Gbif user password.')
    parser.add_argument(
        'gbif_mail', type=str,
        help='Gbif user email.')
    parser.add_argument(
        '--samples_per_cls', type=natural_int, default=300,
        help='Number of samples to download per each specie.')
    parser.add_argument(
        '--num_workers', type=natural_int, default=30,
        help='Number of parallel download processes.')
    parser.add_argument(
        '--num_gbif_requests', type=natural_int, default=3,
        help=('Number of parallel gbif requests. '
              '3 is a maximum available value'))
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(species_file=args.species_file, save_pth=args.save_pth,
         samples_per_cls=args.samples_per_cls, num_workers=args.num_workers,
         num_gbif_requests=args.num_gbif_requests, gbif_name=args.gbif_name,
         gbif_pwd=args.gbif_pwd, gbif_mail=args.gbif_mail)
