import os
import pdb
import time
import argparse
import subprocess
import sys

from loguru import logger

import yaml

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dataclasses import asdict

from src import Config, CLASSIFICATION_LOSSES
from src import setup_datasets, setup_dataloaders, setup_model, setup_optimizer

try:
    import wandb
except ImportError:
    wandb = None

def setup_logger(args: Config):
    args.host_name = subprocess.run(
        'hostname', shell=True, check=True,
        stdout=subprocess.PIPE).stdout.decode('utf-8').strip()

    output_dir = args.output_dir
    if output_dir is not None and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not args.log_to_stdout:
        logger.remove()
    if output_dir is not None:
        logger.add(os.path.join(args.output_dir, 'training.log'))

    return output_dir


def setup_wandb(args: Config):
    if args.wandb:
        assert wandb is not None, 'Please install wandb'
        logger.debug('Starting wandb.')
        name = "+".join(args.wandb_tags) if args.wandb_run_name is None else args.wandb_run_name
        wandb.init(
            project=args.wandb_project_name,
            name=name,
            id=wandb.util.generate_id(),
            tags=args.wandb_tags,
            config=vars(args),
            dir=args.output_dir
        )
        logger.debug('Finished loading wandb.')


# ----------------------------- SINGLE DATALOADER PASS ---------------------------------
def run_batches(mode_name, is_training, args: Config, loader, model,
                optimizer, averager, lr_scheduler, seen_examples, total_examples):
    is_classification = args.loss in CLASSIFICATION_LOSSES
    model.train(is_training)

    per_batch_metrics = []

    cum_loss = 0.0
    cum_examples = 0.0

    cum_correct = 0.0

    start_time = time.time()
    data_start_time = time.time()
    elapsed_data_time = 0
    num_training_steps = 0
    take_training_step = False

    for batch_idx, (x, y) in enumerate(loader):
        elapsed_data_time += time.time() - data_start_time

        x, y = x.to(args.device), y.to(args.device)

        with torch.set_grad_enabled(is_training):
            outputs = model(x)
            if args.loss in ('log', 'xentropy'):
                loss = F.cross_entropy(outputs, y)
            elif args.loss in ('square', 'mse'):
                loss = F.mse_loss(outputs, y)

        if is_training:
            if batch_idx % args.grad_accum_steps == 0:
                optimizer.zero_grad()
            (loss / args.grad_accum_steps).backward()
            take_training_step = (batch_idx + 1) % args.grad_accum_steps == 0
            if take_training_step:
                num_training_steps += 1
                optimizer.step()
                averager.step()
                for _ in range(args.grad_accum_steps):
                    lr_scheduler.step()  # todo: fix scheduler defs so we don't have to call step grad_accum_steps times
            seen_examples += x.shape[0]

        cum_loss += loss.item() * x.shape[0]
        cum_examples += x.shape[0]

        if is_classification:
            pred = outputs.argmax(1)
            num_correct = (pred == y.view_as(pred)
                           ).float().sum().item()
            cum_correct += num_correct

        # TODO: optionally log also distance from init and magnitude of the gradient
        if is_training:
            batch_metrics = dict(
                seen_examples=seen_examples,
                loss=loss.item(),
                batch_size=x.shape[0],
                learning_rate=float(optimizer.param_groups[0]['lr']),
                accuracy=num_correct / x.shape[0]
            )
            per_batch_metrics.append(batch_metrics)
            if args.wandb:
                wandb.log({'train/batch/' + k: v for k, v in batch_metrics.items()})

        if is_training:
            add_to_log = (take_training_step and (
                    num_training_steps == 1 or num_training_steps % args.log_interval == 0)
                          ) or batch_idx == len(loader) - 1
        else:
            add_to_log = (args.log_interval_eval and (batch_idx > 0 and batch_idx % args.log_interval_eval == 0)) or int(cum_examples) == len(loader.dataset)

        if add_to_log:
            log_str = (f'{mode_name} at {seen_examples}/{total_examples} '
                       f'({100 * seen_examples / total_examples:.1f}%); '
                       f'Loss={cum_loss / cum_examples:.4g}')
            if is_classification:
                log_str += f', Acc={cum_correct / cum_examples * 100: .2f}% ({int(cum_correct)} / {int(cum_examples)})'
            logger.info(log_str)

        data_start_time = time.time()

    elapsed_time = time.time() - start_time

    run_metrics = dict(
        loss=cum_loss / cum_examples,
        accuracy=cum_correct / cum_examples,
        time=elapsed_time,
        data_time=elapsed_data_time,
    )
    if is_training:
        run_metrics['seen_examples'] = seen_examples
        run_metrics['learning_rate'] = per_batch_metrics[-1]['learning_rate']

    return run_metrics, per_batch_metrics, seen_examples


# -------------------------------- MAIN LOOP -----------------------------------
# @logger.catch(onerror=lambda _: sys.exit(1))
@logger.catch(reraise=True)
def experiment_generator(args: Config):
    output_dir = setup_logger(args)
    setup_wandb(args)
    save_output = args.output_dir is not None
    save_checkpoints = save_output and args.save_freq > 0
    save_best_checkpoint = save_output and args.save_best_checkpoint
    patience = args.early_stopping_patience

    datasets = setup_datasets(args)
    model = setup_model(datasets['train'], args)
    loaders = setup_dataloaders(datasets, model, args)

    if args.seed_training is not None:  # re-seed the rng so that we get the same base_lr for each seed regardless of the model architecture
        torch.manual_seed(args.seed_training)
    optimizer, averager, lr_scheduler, num_batches = setup_optimizer(args, model)

    logger.info(f'Args:\n{yaml.dump(asdict(args), sort_keys=True)}')
    if output_dir is not None:
        config_path = os.path.join(output_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(asdict(args), f, sort_keys=True)
            if args.wandb:
                wandb.save(config_path)

    stats_df = pd.DataFrame()

    seen_examples = 0

    best_metric = 0.0

    non_train_splits = [split for split in datasets if split != 'train']

    # pdb.set_trace()

    # TODO: recover from saved checkpoint!
    if args.steps_per_epoch is None:
        num_epochs = int(np.round(num_batches / (args.data_n_examples['train'] / args.batch_size_accum)))
    else:
        num_epochs = num_batches // (args.steps_per_epoch * args.grad_accum_steps)
    total_examples = num_epochs * len(loaders['train']) * args.batch_size_accum  # exact assuming we drop non-full batches

    try:
        for epoch in range(1, num_epochs + 1):
            logger.info(120 * '=')
            logger.info('Starting epoch %d / %d' % (epoch, num_epochs))

            train_stats, _, seen_examples = run_batches(
                'TRAIN', True, args, loaders['train'], model, optimizer, averager, lr_scheduler,
                seen_examples, total_examples)
            stats = {'train/' + k: v for k, v in train_stats.items()}

            logger.info(120 * '=')
            models_to_eval = dict(last=model)
            if averager.method != 'none':
                models_to_eval['av'] = averager.averaged_model
            for split_name in non_train_splits:
                loader_eval = loaders[split_name]
                for model_name, model_to_eval in models_to_eval.items():
                    eval_stats, _, seen_examples = run_batches(
                        f'EVAl {split_name.upper()} {model_name.upper()}', False, args,
                        loader_eval, model_to_eval, None, None, None,
                        seen_examples, total_examples
                    )
                    stats.update({split_name + '/' + model_name + '/' + k: v
                                  for k, v in eval_stats.items()})

            stats_df = pd.concat([stats_df, pd.DataFrame([stats], index=[0])],
                                 ignore_index=True)
            if args.wandb:
                wandb.log(stats)

            if save_output:
                stats_df.to_csv(os.path.join(output_dir, 'stats.csv'))

            # save checkpoint
            if save_checkpoints and (
                    epoch % args.save_freq == 0 or epoch == num_epochs):
                for model_name, model_to_eval in models_to_eval.items():
                    torch.save(dict(epoch=epoch,
                                    config=asdict(args),
                                    checkpoint_type=model_name,
                                    state_dict=model_to_eval.state_dict()),
                               os.path.join(output_dir,
                                            f'checkpoint-{model_name}-epoch{epoch}.pt'))
                    torch.save(dict(epoch=epoch,
                                    config=asdict(args),
                                    state_dict=optimizer.state_dict()),
                               os.path.join(output_dir,
                                            f'optimizer-checkpoint-epoch{epoch}.pt'))

            if save_best_checkpoint:
                metrics = stats_df.filter(regex=args.best_checkpoint_metric)
                last_metrics = metrics.iloc[-1]
                last_metric, last_metric_name = last_metrics.max(), last_metrics.idxmax()
                if last_metric > best_metric:
                    best_metric = last_metric
                    model_to_save = model if 'last' in last_metric_name else averager.averaged_model
                    output_path = os.path.join(output_dir, f'best_checkpoint.pt')
                    logger.info(f'{last_metric_name} set a new record of {last_metric:.2%}, saving to {output_path}')
                    save_dict = dict(metric=best_metric, metric_name=last_metric_name, epoch=epoch, config=asdict(args),
                                     state_dict=model_to_save.state_dict())
                    torch.save(save_dict, output_path)

            yield stats_df, model

            loss_values = stats_df.filter(regex='loss')
            if np.any(loss_values.isna().values) or np.max(loss_values.values) > args.max_allowed_loss:
                logger.error(
                    'Loss exceeded threshold value; breaking training')
                break

            if patience is not None:
                metrics = stats_df.filter(regex=args.best_checkpoint_metric)
                not_ok = metrics.iloc[-patience:].max() < metrics.max()
                if not_ok.all():
                    logger.info(f'No improvement in the last {patience} epochs - stopping the training')
                    break

        if save_output:
            with open(os.path.join(output_dir, 'done'), 'w'):
                pass
    except KeyboardInterrupt:
        pass


def run_experiment(args):
    for stats_df, model in experiment_generator(args):
        pass
    return stats_df, model, args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str,
                        help='path to YAML file containing job configuration')

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    return Config(**config)


if __name__ == '__main__':
    _ = run_experiment(parse_args())
