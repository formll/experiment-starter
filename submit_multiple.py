import yaml
import os
import shutil
import itertools
import argparse
import subprocess
import datetime
import pandas as pd
import numpy as np
from copy import deepcopy
from time import sleep

try:
    import wandb
except ImportError:
    wandb = None


def format_key(k):
    return '_'.join([kk[:3] for kk in k.split('_')])


def format_value(v):
    return str(v).replace(' ', '').replace('/', '-')


def unwind_grid(grid_dict):
    list_keys = [k for k in grid_dict.keys() if k.startswith('LIST')]
    grid = []
    for k in list_keys:
        subgrid_list = grid_dict.pop(k)
        subgrid = []
        for subgrid_dict in subgrid_list:
            subgrid.extend([tuple(zip(*el)) for el in unwind_grid(subgrid_dict)])
        grid.append(tuple(set(subgrid)))
    for k, v in grid_dict.items():
        if isinstance(v, str):
            v = list(eval(v))
        if not isinstance(v, list) or isinstance(v, tuple):
            v = [v]
        grid.append(tuple((k, vv) for vv in v))
    return list(itertools.product(*grid))


def grid_to_str(list_of_dicts):
    nunique = pd.DataFrame(list_of_dicts).nunique(dropna=False)
    nunique = nunique[nunique > 1]
    assert nunique.prod() == len(list_of_dicts)
    return ' x '.join(f'{n} {k}' for k, n in nunique.items())


def expand_tuple_keys(dict_with_tuples):
    tuple_keys = [k for k in dict_with_tuples if isinstance(k, tuple)]
    if len(tuple_keys) == 0:
        return dict_with_tuples
    else:
        for kk in tuple_keys:
            vv = dict_with_tuples.pop(kk)
            dict_with_tuples.update(dict(zip(kk, vv)))
        return expand_tuple_keys(dict_with_tuples)


def get_wandb_args(job_details, wandb_tags, wandb_run_name):
    wandb_args = dict()
    if job_details['wandb']:
        wandb_args = dict(
            wandb=job_details['wandb'],
            wandb_tags=wandb_tags,
            wandb_run_name=wandb_run_name,
            wandb_project_name=job_details['wandb_project_name'])
    return wandb_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('jobfile', type=str,
                        help='path to YAML file containing job configuration')
    parser.add_argument('-s', '--script', type=str, help='name of script to provide to sbatch commands')
    parser.add_argument('-y', '--yes', action='store_true', help='confirm submission without viewing grid details first')
    parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite existing results directory')
    parser.add_argument('-d', '--dry_run', action='store_true',
                        help='prints the sbatch commands instead of executing them, and does not create an output folder')
    parser.add_argument('-r', '--rerun', action='store_true',
                        help='re-submits all jobs not marked as done; when used the jobfile argument should be the path '
                             'to the results folder of a previous execution of the script.')

    args = parser.parse_args()

    run_str = 'RUN' if not args.rerun else 'RE-RUN'
    if args.dry_run:
        print(f'===== THIS IS A ***DRY*** {run_str} =====')
    else:
        print(f'===== THIS IS A ***REAL*** {run_str} =====')

    if not args.rerun:
        with open(args.jobfile, 'r') as f:
            job_description = yaml.safe_load(f)

        job_details = job_description['job_details']
        output_dir = job_details['output_dir']
        batch_name = datetime.datetime.now().strftime('%y-%m-%d') + '-' + job_details['name']
        batch_dir = os.path.join(output_dir, batch_name)
    else:
        jobfile = os.path.join(args.jobfile, 'job.yaml')
        with open(jobfile, 'r') as f:
            job_description = yaml.safe_load(f)
        job_details = job_description['job_details']
        batch_dir = args.jobfile
        batch_name = os.path.split(batch_dir)[-1]

    if job_details['wandb']:
        wandb.login()

    if os.path.exists(batch_dir) and os.listdir(batch_dir):
        if not (args.overwrite or args.rerun):
            raise FileExistsError('Directory exists and overwrite flag is not set')
        if args.overwrite:
            print('Removing existing output directory')
            shutil.rmtree(batch_dir)

    print(f'Writing experiment result to directory {batch_dir}')

    # --------------------- HANDLE CONFIG ENUMERATION------------------------
    configs = [dict(c) for c in unwind_grid(deepcopy(job_description['parameters']))]
    configs_summary_str = grid_to_str(configs)
    configs = list(map(expand_tuple_keys, configs))
    total_configs = len(configs)
    nunique_keys = pd.DataFrame(configs).nunique(dropna=False)
    varying_keys = nunique_keys[nunique_keys > 1].index.values

    print(f'Ready to run {total_configs} configurations: {configs_summary_str}')
    if not args.yes:
        input("Press Enter to continue...")

    def grid_generator():
        for i, kvs in enumerate(configs):
            varying_specs = [f'{format_key(k)}={format_value(kvs[k])}' for k in varying_keys]
            spec_name = '+'.join(varying_specs)
            wandb_run_name = f"{batch_name}/{i:03d}_{spec_name}"
            spec_name = f'{i:03d}_{batch_name}+{spec_name}'
            out_dir = os.path.join(batch_dir, spec_name)
            if os.path.exists(os.path.join(out_dir, 'done')):
                continue
            spec = dict(output_dir=out_dir, 
                        **get_wandb_args(job_details, varying_specs, wandb_run_name),
                        **kvs)
            spec_filename = os.path.join(out_dir, 'spec.yaml')
            if not args.dry_run:
                os.makedirs(out_dir, exist_ok=True)
                with open(spec_filename, 'w') as f:
                    yaml.dump(spec, f, sort_keys=True)

            yield spec_name, spec_filename, out_dir

    def grid_generator_rerun():
        for spec_name in os.listdir(batch_dir):
            out_dir = os.path.join(batch_dir, spec_name)
            if not os.path.isdir(out_dir) or spec_name.startswith('.'):
                continue
            if os.path.exists(os.path.join(out_dir, 'done')):
                continue
            spec_filename = os.path.join(out_dir, 'spec.yaml')
            yield spec_name, spec_filename, out_dir


    # --------------------- HANDLE OUTPUT FOLDER -------------------------------
    if not (args.dry_run or args.rerun):
        os.makedirs(batch_dir, exist_ok=True)
        job_outfile = os.path.join(batch_dir, 'job.yaml')
        if os.path.exists(job_outfile):
            c = 1
            while os.path.exists(os.path.join(batch_dir, f'job.old.{c}.yaml')):
                c += 1
            os.rename(job_outfile,
                      os.path.join(batch_dir, f'job.old.{c}.yaml'))
            print(f'Renaming old job file to job.old.{c}.yaml')

        with open(job_outfile, 'w') as f:
            yaml.dump(job_description, f, default_flow_style=False)

    count = 0
    gen = grid_generator if not args.rerun else grid_generator_rerun
    for spec_name, spec_filename, out_dir in gen():
        out_path = os.path.join(out_dir, '%j.out')
        cmd = f'sbatch --job-name={spec_name} --output={out_path} --error={out_path} {args.script} {spec_filename}'
        if args.dry_run:
            print(f'Would now run "{cmd}"')
            if args.rerun:
                with open(os.path.join(out_dir, 'config.yaml'), 'r') as f:
                    prev_config = yaml.safe_load(f)
                print(f'Host of previous failed run: {prev_config["host_name"]}')
        else:
            while True:
                try:
                    sbatch_output = subprocess.run(cmd, shell=True, check=True)
                    break
                except subprocess.CalledProcessError:
                    print('Encountered called process error while submitting, waiting and trying again')
                    sleep(4 + float(np.random.rand(1) * 5))
        count += 1

    print(f'\nStarted {count} jobs in total')




