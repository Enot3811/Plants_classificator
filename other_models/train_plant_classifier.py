"""Train plant classificator."""

from pathlib import Path
import shutil
import json
import argparse
import sys
import csv

import torch
import torch.optim as optim
from torch import FloatTensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Metric, MetricCollection, ClasswiseWrapper
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassAUROC)
from tqdm import tqdm
import albumentations as A
import pandas as pd

sys.path.append(str(Path(__file__).parents[1]))
from dataset_utils.plants_dataset import PlantsDataset
from other_models.models_factory import ModelFactory


class LossMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('loss',
                       default=torch.tensor(0, dtype=torch.float64),
                       dist_reduce_fx='sum')
        self.add_state('n_total',
                       default=torch.tensor(0, dtype=torch.float64),
                       dist_reduce_fx='sum')

    def update(self, batch_loss: FloatTensor):
        self.loss += batch_loss
        self.n_total += 1

    def compute(self):
        return self.loss / self.n_total


def main(config_pth: Path):
    # Read config
    with open(config_pth, 'r') as f:
        config_str = f.read()
    config = json.loads(config_str)

    # Prepare some stuff
    torch.manual_seed(config['random_seed'])
    torch.multiprocessing.set_start_method('spawn')

    if config['device'] == 'auto':
        device: torch.device = (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu'))
    else:
        device: torch.device = torch.device(config['device'])

    metrics_dict = {
        'accuracy': MulticlassAccuracy, 'precision': MulticlassPrecision,
        'recall': MulticlassRecall, 'auroc': MulticlassAUROC}
    
    work_dir = Path(config['work_dir'])
    tensorboard_dir = work_dir / 'tensorboard'
    ckpt_dir = work_dir / 'ckpts'
    metrics_log_dir = work_dir / 'metrics_log'
    if not config['continue_training']:
        if work_dir.exists():
            input(f'Specified directory "{str(work_dir)}" already exists. '
                  'Сontinuing to work will delete the data located there. '
                  'Press enter to continue.')
            shutil.rmtree(work_dir)
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        metrics_log_dir.mkdir(parents=True, exist_ok=True)

    # Check and load checkpoint
    if config['continue_training']:
        checkpoint = torch.load(ckpt_dir / 'last_checkpoint.pth')
        model_params = checkpoint['model_state_dict']
        optim_params = checkpoint['optimizer_state_dict']
        lr_params = checkpoint['scheduler_state_dict']
        start_ep = checkpoint['epoch']
    else:
        if config['pretrained']:
            model_params = torch.load(config['pretrained'])['model_state_dict']
        else:
            model_params = None
        optim_params = None
        lr_params = None
        start_ep = 0

    # Get tensorboard
    log_writer = SummaryWriter(str(tensorboard_dir))

    # Get transforms
    train_transforms = [
        A.RandomResizedCrop(*config['result_size'], scale=(0.8, 1.0))]
    if config['train_transforms']['horizontal_flip']:
        train_transforms.append(A.HorizontalFlip())
    if config['train_transforms']['vertical_flip']:
        train_transforms.append(A.VerticalFlip())
    if config['train_transforms']['blur']:
        train_transforms.append(A.Blur(blur_limit=3, p=0.5))
    if config['train_transforms']['color_jitter']:
        train_transforms.append(A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0, p=0.2))
    train_transforms = A.Compose(train_transforms)
    
    val_transforms = [
        A.RandomResizedCrop(*config['result_size'], scale=(0.8, 1.0))]
    if config['val_transforms']['horizontal_flip']:
        val_transforms.append(A.HorizontalFlip())
    if config['val_transforms']['vertical_flip']:
        val_transforms.append(A.VerticalFlip())
    if config['val_transforms']['blur']:
        val_transforms.append(A.Blur(blur_limit=3, p=0.5))
    if config['val_transforms']['color_jitter']:
        val_transforms.append(A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0, p=0.2))
    val_transforms = A.Compose(val_transforms)

    # Get dataset's helper data
    with open(Path(config['dataset']) / config['class_info']) as f:
        cls_weights = []  # for loss
        cls_metric_names = []  # for classwise wrappers
        cls_to_id = {}  # for dataset
        for cls_id, (cls_name, _, cls_weight) in enumerate(csv.reader(f)):
            cls_weights.append(float(cls_weight))
            cls_to_id[cls_name] = cls_id
            cls_metric_names.append(f'{cls_name} {cls_id}')
    cls_weights = torch.tensor(
        cls_weights, dtype=torch.float32, device=device)
    num_classes = len(cls_to_id)

    # Get datasets and loaders
    train_dir = Path(config['dataset']) / 'train'
    val_dir = Path(config['dataset']) / 'val'

    train_dset = PlantsDataset(
        train_dir, device=device, transforms=train_transforms,
        class_to_index=cls_to_id)
    val_dset = PlantsDataset(
        val_dir, device=device, transforms=val_transforms,
        class_to_index=cls_to_id)

    train_loader = DataLoader(train_dset,
                              batch_size=config['batch_size'],
                              shuffle=config['shuffle_train'],
                              collate_fn=PlantsDataset.collate_func,
                              num_workers=config['num_workers'])
    val_loader = DataLoader(val_dset,
                            batch_size=config['batch_size'],
                            shuffle=config['shuffle_val'],
                            collate_fn=PlantsDataset.collate_func,
                            num_workers=config['num_workers'])

    # Get the model
    model = ModelFactory.create_model_by_name(
        config['architecture'], num_classes=num_classes)
    model.to(device=device)
    if model_params:
        model.load_state_dict(model_params)

    # Get the loss function
    loss_function = torch.nn.CrossEntropyLoss(
        weight=cls_weights).to(device=device)

    # Get the optimizer and AMP scaler
    optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                           weight_decay=config['weight_decay'])
    if optim_params:
        optimizer.load_state_dict(optim_params)
    amp_scaler = torch.cuda.amp.grad_scaler.GradScaler(
        enabled=config['use_amp'])

    # Get the scheduler
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['n_epoch'], eta_min=config['min_lr'],
        last_epoch=start_ep - 1)
    if lr_params:
        lr_scheduler.load_state_dict(lr_params)
    
    # Get metrics
    train_metrics = {
        metric_name: ClasswiseWrapper(
            metric_function(num_classes, average=None, compute_on_cpu=True),
            labels=cls_metric_names, prefix=metric_name + ' ')
        for metric_name, metric_function in metrics_dict.items()
    }
    train_metrics = MetricCollection(train_metrics).to(device=device)
    train_loss_metric = LossMetric().to(device=device)
    
    val_metrics = {
        metric_name: ClasswiseWrapper(
            metric_function(num_classes, average=None, compute_on_cpu=True),
            labels=cls_metric_names, prefix=metric_name + ' ')
        for metric_name, metric_function in metrics_dict.items()
    }
    val_metrics = MetricCollection(val_metrics).to(device=device)
    val_loss_metric = LossMetric().to(device=device)

    # Do training
    best_metric = None
    for epoch in range(start_ep, config['n_epoch']):

        print(f'Epoch {epoch + 1}')

        # Train epoch
        model.train()
        step = 0
        for batch in tqdm(train_loader, 'Train step'):
            # Do forward and backward
            with torch.autocast(device_type=str(device), dtype=torch.float16,
                                enabled=config['use_amp']):
                images, targets, _ = batch
                images = images.to(device=device)
                targets = targets.to(device=device)
                predicts = model(images)
                loss = loss_function(predicts, targets)
            amp_scaler.scale(loss).backward()

            # Update metrics
            train_metrics.update(
                preds=predicts.detach(), target=targets.detach())
            train_loss_metric.update(batch_loss=loss.detach())

            # Check step
            if ((step + 1) % config['n_accumulate_steps'] == 0 or
                    (step + 1) == len(train_loader)):
                # make weights update only on certain steps
                amp_scaler.step(optimizer)
                amp_scaler.update()
                optimizer.zero_grad()
            step += 1

        # Val epoch
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, 'Val step'):
                images, targets, _ = batch
                images = images.to(device=device)
                targets = targets.to(device=device)
                predicts = model(images)
                loss = loss_function(predicts, targets)

                # Update metrics
                val_metrics.update(
                    preds=predicts.detach(), target=targets.detach())
                val_loss_metric.update(batch_loss=loss)

        # Lr scheduler
        lr_scheduler.step()
        lr = lr_scheduler.get_last_lr()[0]

        # Compute epoch metrics
        metrics = {'train': {}, 'val': {}}
        metrics['train']['metrics'] = train_metrics.compute()
        metrics['train']['loss'] = train_loss_metric.compute()
        metrics['val']['metrics'] = val_metrics.compute()
        metrics['val']['loss'] = val_loss_metric.compute()

        train_metrics.reset()
        train_loss_metric.reset()
        val_metrics.reset()
        val_loss_metric.reset()

        # Parse metrics
        # Metric name is "{metric} {specie1} {specie2} {cls_id}"
        # For example "accuracy Dubouzetia confusa 0"

        for cls_name, cls_id in cls_to_id.items():
            cls_metrics = {}
            # Select metrics belonging to one class
            # and write them to corresponding file
            
            for mode in ['train', 'val']:
                # Get full metric names for this class
                selected_names = filter(
                    lambda metric_name: int(metric_name.split()[-1]) == cls_id,
                    metrics[mode]['metrics'])
                # Make a dict when keys are metrics names like "accuracy" etc
                # and values are one element lists with float
                cls_metrics.update({
                    mode + ' ' + metric_name.split()[0]: [
                        metrics[mode]['metrics'][metric_name].item()]
                    for metric_name in selected_names})
                # Add loss
                cls_metrics[f'{mode} loss'] = metrics[mode]['loss'].item()
                # Pack to DataFrame for convenient writing
                frame_to_write = pd.DataFrame(cls_metrics)
            # Write to class csv file
            file_pth = metrics_log_dir / f'{cls_id}_{cls_name}.csv'
            if file_pth.exists():
                kwargs = {
                    'mode': 'a',
                    'header': False}
            else:
                kwargs = {'mode': 'w'}
            kwargs['index'] = False
            frame_to_write.to_csv(file_pth, **kwargs)

        # Select metrics type wisely for tensorboard
        for metric_name in metrics_dict:
            selected_metrics = {}
            # Get some metric for train and val and store it to dict
            for mode in ['train', 'val']:
            
                # Get full metric names for this metric type
                selected_metrics_names = filter(
                    lambda selected_metric: (
                        selected_metric.split()[0] == metric_name),
                    metrics[mode]['metrics'])
                # Get values by selected names and get mean
                selected_metrics[mode] = torch.tensor([
                    metrics[mode]['metrics'][selected_metric]
                    for selected_metric in selected_metrics_names]).mean()
            # Write metric to tensorboard
            log_writer.add_scalars(metric_name, selected_metrics, epoch)

        # Add loss separately
        log_writer.add_scalars('loss', {
            'train': metrics['train']['loss'],
            'val': metrics['val']['loss']
        }, epoch)

        log_writer.add_scalar('lr', lr, epoch)

        print(f'Train Loss: {metrics["train"]["loss"].item()}')
        print(f'Val Loss: {metrics["val"]["loss"].item()}')
        print('Lr:', lr)

        # Save model
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'epoch': epoch + 1
        }
        torch.save(checkpoint, ckpt_dir / 'last_checkpoint.pth')

        if (best_metric is None or best_metric > metrics['val']['loss']):
            torch.save(checkpoint, ckpt_dir / 'best_checkpoint.pth')
            best_metric = metrics['val']['loss']

    log_writer.close()


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'config_pth', type=Path,
        help='Path to train config file.')
    args = parser.parse_args()

    if not args.config_pth.exists():
        raise FileNotFoundError('Specified config file does not exists.')
    return args


if __name__ == "__main__":
    args = parse_args()
    config_pth = args.config_pth
    main(config_pth=config_pth)
