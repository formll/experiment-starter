# Experiment starter code

The goal of this repository is to help you quickly start your ML/optimization experiments by providing a 
fairly generic and extensible scripts for pytorch-based training + slurm submission automation. Key features include:
- Integration of multiple libraries for data and model management (and particularly `timm`) 
- Strong reproducibility via flexible configuration architecture
- Robust logging functionality enabling a bird's eye view of many experiments combined (see 
- ["Analyzing experiment results"](#analyzing-experiment-results) below)
- Ability to define batches of experiments and automatically submit them all to slurm with one script

__Intended use.__ Basically - make a copy and modify it in whatever way you see fit! Since experiments are all about
trying new and exciting things, every new experiment will likely need to add new, special-purpose functionality 
to this code, and possibly also break some existing functionality. The code here is intended as a general guideline 
for good configurability, logging and automation practices - it is not intended to force you into any code structure or
external library that doesn't work for your project.


__Note.__ While this script is born out of many past researcher projects, in its current form it is still fairly new 
and only briefly tested - you should expect some sharp corners and even an occasional bug. There are also many 
important features that are currently missing - see
["Contributing to this repository"](#contributing-to-this-repository) and ["TODO's"](#todos) below. Finally, the 
internals of the code could certainly use more documentation; this will happen some day (hopefully).


## Setting up

The following provides instructions for getting started with the code to the point you can run it on TAU machines
(specifically the `c-00x` nodes and the slurm cluster). For more info on getting access to TAU machine, 
talk to someone in the group.

### Getting the code

When using this code to start your own experiment, you do not want to clone the repository. Instead, you should fork 
it and give a new and appropriate name. I suggest forking the repository by pressing the "fork" button in github
and then cloning the forked remote repository. Also, I strongly recommend first getting the code into your personal 
computer and then syncing it to TAU using the remote deployment feature in a modern IDE (see [best practices](#best-practices-and-general-cotchas))
below.

### Conda environment on TAU machines

Before installing the conda environment for the project (or really any environment at all!) you should make sure that 
all libraries and temporary files are stored in the group's network drive and not your (tiny) home directory. To do that,
first move the `.cache` folder to the network storage as follows:
```shell
rm -rf ~/.cache # or mv it if it contains something you wish to keep
mkdir /home/ycarmon/users/${USER}/.cache
ln -s /home/ycarmon/users/${USER}/.cache ~/
```

Next, [install miniconda](https://docs.conda.io/en/latest/miniconda.html) to  `/home/ycarmon/users/${USER}/` 
(and *not* your home directory!).

Finally, from the project's main directory run:
```shell
conda env create -f environment.yml
```
to install all the dependencies. This installation takes a while (up to 30 minutes is reasonable) because some of the 
dependencies (mainly PyTorch and Tensorflow) are heavy. After the environment is installed, don't forget to activate it 
with
```shell
conda activate dev
```

Note that while this project uses conda for environment management, all package management is actual done using pip, 
because in my experience it is faster and less error-prone.

### Creating symlinks for data and output folders
You absolutely don't want to store data and checkpoints in your home folder. You also don't want to download datasets 
already present on our storage. Moreover, you don't want to save your (possibly large) checkpoints and experiments logs
to backed-up parts of our storage. The following commands take care of all of this, where `${PROJECT_FOLDER}` is the 
name of the folder you chose for your experiments.

```shell
cd ~/${PROJECT_FOLDER}
ln -s /home/ycarmon/data .
mkdir /home/ycarmon/no_backup/users/${USER}/${PROJECT_FOLDER}/results
ln -s /home/ycarmon/no_backup/users/${USER}/${PROJECT_FOLDER}/results .
```

## Code structure

The repository contains two main scripts:

1. __`train.py`__ \
Performs a single experiment (= model training run) with configuration specified by a YaML file
(for example `test_cfg_covtype_linear.yaml`). __To learn about the different configurations (and see their default values) 
take a look at `src/config.py`__. In your experiments, you would likely make multiple changes to `train.py` and its dependencies.
2. __`submit_multiple.py`__\
Submits multiple slurm jobs, each consisting of one experiment (i.e., a call to `train.py`). To do
this, the script takes in a "job specification" in the form of a YaML file (for example `jobs/linear-classifier-example.yaml`)
which defines a grid of inputs to `train.py`. The `submit_multiple.py` script then creates a folder (whose name is 
based on the job specification) such that each subfolder contains a file called `spec.yaml` which is the input to
`train.py` for a particular experiment. Then, the script runs the command `sbatch {--some args} submit.sh {job_dir}/{experiment_dir}/spec.yaml`
where
   * `{--some args}` set the slurm job name to be the experiment name, and direct to the slurm output to the directory 
   `{job_dir}/{experiment_dir}`.
   * `submit_script.sh` is a short script that defines all other slurm configuration, including the amount and
   type of requested GPU, slurm partition, amount of requested memory, etc. The repository comes with a few example 
   scripts: `submit_cpu.sh`, `submit_2080.sh` and `submit_3090plus.sh`; take a look at each script to see what it does.

   In your experiments, it is likely that you would not need to modify `submit_multiple.py` at all, but you will 
   probably need to create some customized `submit.sh` scripts.

The code internals are in the `src` directory. To explore it, start with reading `train.py`.


### Some useful training script features
(partial list)

* Supports both cpu and gpu training. The latter can use multiple gpu's on a single node using the pytorch DataParallel wrapper.
* Provides access to all (potentially pretrained) models in [`timm`](https://github.com/rwightman/pytorch-image-models), 
  as well as CIFAR10-scale models [taken from here](https://github.com/meliketoy/wide-resnet.pytorch) and 
  [OpenAI's CLIP models](https://github.com/openai/CLIP). Linear models are also supported :).
* Provides access to many datasets via [`tfds`](https://www.tensorflow.org/datasets/api_docs/python/tfds)
  and the standard `torchvision` datasets (both via `timm`), as well as classical `libsvm` dataset (via `libsvmdata`).
* Allows to flexibly define the evaluation interval (i.e., "epoch") as either a full data pass (standard), or fixed 
  number of optimization steps / training sample seen (not standard, but very useful).
* Allows flexible specification of evaluation/test splits (particularly strong when combined with `tfds`).
* Automatically infers several model / data dependent configurations such as dataset size, number of classes and input 
  shape. All the inferred info is written back to the configuration and then saved in `config.yaml` in the scripts 
  output folder.
* Comprehensive logging both in free text (via `loguru`; both to a file and optionally to stdout) and in structured csv
  form (via `pandas`).
* Supports gradient accumulation for batch sizes that can't fit in memory.
* Supports early stopping based on a patience heuristic, with flexible configuration of the metric used to determine stopping.
* Supports model averaging and evaluation of both the averaged and last checkpoints.

### The "job specification" format
The job specification yaml is a dict with keys `job_details` and `parameters`. The `job_details` field just defines a name for 
the job (to which `submit_multiple.py` adds a date in the beginning) and the root directory for results; the 
specification itself is in the `parameters` field.  That field is essentially a dictionary of lists, where keys correspond
to configuration name - the grid of experiment configurations is the cartesian product of these lists. However, keys whose
name begin with 'LIST' are special. These keys contain a list of "job specification" dicts and what `submit_multiple` does 
is first expand those to a list (by the cartesian product rule) and then apply the cartesian product on the outer 
specification. Having 'LIST' keys allows us to define essentially any set of experiment configurations. It might sound
complicated, but it really isn't - looking at the example files in `jobs/` and running one of the example should make 
things very clear. If not, just talk to me.



### FAQ
* **Why do job specification files list `data_splits_map` as a string?**\
  A: `submit_multiple` can only handle hashable config, so `train.py` accepts either a dict or a str, 
  and uses eval to convert the latter to the former.
* **Why is the main loop in train.py written as a generator?**\
  A: To allow other python programs to import `experiment_generator` and then perform their own evaluation (or other 
  logic) in between epochs.

## Running the code

Don't forget to `conda activate dev`!
### Single experiment run
Example command line for running a single experiment (mainly useful for debugging):
```shell
python train.py test_cfg_covtype_linear.yaml
```
For scripts that run on gpu, don't forget to set the `CUDA_VISIBLE_DEVICES` environment so that it contains the list 
of gpu(s) visible to your script. For example
```shell
CUDA_VISIBLE_DEVICES=1,3 python train.py test_cfg_cifar10_lenet.yaml
```
You can use 
`nvidia-smi -l 1` to see which GPU is free and monitor their utilization.

### Submitting a batch of experiments to slurm
Running the command
```shell
python submit_multiple.py jobs/linear-classifier-example.yaml -s submit_cpu.sh
```
will submit a grid of 108 cpu jobs. Feel free to give it a try, since we have basically unlimited cpu's on the
`cpu-killable` partition and they are almost never used. You can even go ahead and submit another job 
(`linear-classifier-example-2.yaml`). Once jobs are submitted, you can monitor them by running `squeue --me`.

To run a grid of 4 jobs on our lower-end GPU's you can run
```shell
python submit_multiple.py jobs/cifar10-example.yaml -s submit_2080.sh
```

The `submit_multiple` scripts has a few additional features: overwrite mode (`-o`), retry mode (`-r`), and dry 
run mode (`-d`). You can read more about them in the command line help.

## Opening jupyter-lab
After running the expirements, in order to look and analyze the results, you should open jupyter-lab.
1. Run the following command:
```shell
jupyter-lab --port=27615 --ip="0.0.0.0"
```
Change the port to your favorite unique port.
2. Add "cs.tau.ac.il" to the links that shows up, so it will look like: "http://c-001.cs.tau.ac.il:27615/lab"
3. Open it in your browser

## Analyzing experiment results
After running our experiments, we often wish to look at the results. The jupyter notebook `notebooks/view_results.ipynb`
demonstrate my favorite method for doing so. The key points are:
- Read all experiment outputs (`stats.csv`) to a single, big dataframe (canonically called `big_df`), so that every row
  in the data frame represents a different experiment, and each row contains *all* the relevant information about the 
  experiment. More specifically, the set of columns in `big_df` is the union of all configuration keys and all 
  columns in `stats.df`. The values in the columns coming from `stats.csv` are themselves pandas.Series instances, that 
  contain the original columns from `stats_df` corresponding to each experiment.
- Use powerful pandas functions, and particulary `query` and `groupby` to efficiently slice and dice `big_df`.
- Write configurable, flexible and customized visualizations showing what you need. Keep in mind that often the 
  visualization you write will be applied multiple times over many experiments, so it's usually worth taking a bit of 
  time (say an hour or three) to make them flexible and easy to read.

Right now the example notebook contains only a single, not very polished visualization. Future updates will hopefully 
improve on this to add more examples.

Another nice thing that the notebook does for you is include an `rsync` command that lets you sync the remote results 
folder with your local machine, allowing you to run the notebook locally for a smoother user experience.

**What about tensorboard/wandb/comet?** I am not a fan of GUI's for tracking training runs because in my experience
the time invested in messing around with these GUI's eventually exceeds (often by far) the time it takes to create 
the visualizations that shows you exactly what you want. That said, I'm fine with *additionally* integrating one
or more of these tools with the script (it's in the todo's).

## Best practices and general gotchas
- When adding a new functionality it is very important to **set default parameters such that the default behaviour is 
  identical to the behaviour before the functionality was added**. For example, suppose we wanted to add the option to use
  MixUp augmentation. This would involve adding a boolean configuration parameter called something like `use_mixup` - 
  it is then crucial to set the default parameter of `use_mixup` to `False`. The reason that this is important is that 
  for experiments that ran before the parameter was added, we will not have a record of the value of that parameter, and
  we would have to assume the default value. Therefore, if the default value implies a behaviour different from what 
  actually took place in that experiment, we would come to the wrong conclusion. In the previous example, setting the 
  default value of `use_mixup` to `True` would make it look like all past experiments used MixUp, a tragic mix-up indeed!
- The above principle has an important corollary: **once experiments started, never change any default value!**
- One exception to the above rules are configurations that do not affect what the training script does, but only affect
  how logging works.
- Common git usage mistakes such as forgetting to `add` or `push` waste everyone's time, and they are easy to avoid - just
  use a graphical git utility. I think [SourceTree](https://www.sourcetreeapp.com/) is pretty good.
- Do not develop code on remote machine - the machine could go down and you will lose any uncommitted changes, plus you 
  can't use a git GUI on a remote machine. Instead, use the remote deployment feature in a modern IDE (e.g., PyCharm 
  or VS Code).
- [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) is great - you should totally give it a try! It is easy to 
  start jupyter lab on a remote machine (e.g., one of the `c-00x` nodes) and then connect from your laptop. For your,
  convenience, it's included in the `environment.yaml` dependencies.
- Commit and push your code often - a push a day keeps the annoyed collaborator / adviser away...
- [YaML is picky about numbers written in scientific notation](https://github.com/yaml/pyyaml/issues/173). For example, 
  1e3 is interpreted as a string, and so is 1.0e3; but 1.0e+3 is fine. Keep this in mind when specifying your jobs.
- The training script has the ability to download missing datasets and pretrained models. However, if you start multiple
  training scripts in parallel and each attempts to download the same thing at once, a deadlock will occur and likely
  things will break. Therefore, it's better to handle downloads by first making a single run in test mode.
- The current training scripts saves the `stats` dataframe in csv format. This has the great advantage of allowing easy
  browsing (for example, using jupyterlab). However, there is a downside: if the `stats` dataframe contains long numpy 
  array (for example, per-class accuracies), the csv conversion could turn them into strings and ruin them. If you need
  to log something like this, switch from csv to a binary format such as pickle or parquet.

## Contributing to this repository
Want to add features? Fix bugs? Improve the documentation? Great! For that purpose the best thing to do is clone this 
repo, create a branch, and submit a pull request. You should also be able to submit pull requests from forks, but please
make sure to only contribute general-purpose features and not something specific to a particular experiment.

## TODO's

### Big / Medium
- [ ] Support for graceful resume from interruption
- [ ] Support for auto-re-queuing of timed-out jobs
- [ ] Better multi-gpu support via DistributedDataParallel
- [ ] Support for the `compile` feature of PyTorch 2.0
- [ ] Hugging Face integration and NLP capabilities
- [ ] Advanced augmentations (rand augment, mixup, etc.) from timm
- [ ] AMP and other running time optimizations from timm
- [ ] WandB / Comet integration
- [ ] Logging cpu, gpu and memory utilization (note: wandb does that automatically, at least to some extent)
- [ ] Support evaluation on multiple datasets (e.g., ImageNetV2)
- [ ] Sent Slack / SMS notifications for jobs finishing / failing etc. (needs some design to make sure we don't get
      overwhelmed with notifications in jobs with many experiments)
- [ ] Add more types of data analysis and visualizations to `notebooks/view_results.ipynb`
- [ ] Add support for zero-shot training and inference via CLIP
- [ ] Consider using [submitit](https://github.com/facebookincubator/submitit) for the slurm interface (not sure if it's actually helpful)

### Small
- [ ] Make the folder order in `submit_multiple.py` deterministic
- [ ] Improve log formatting
- [ ] Correct early stopping on descending metrics (error, loss, etc.)
- [ ] Make it possible to evaluate only the averaged checkpoint (as opposed to the last)
- [ ] Add script of notebook for generating a report on how many experiments in each job were completed
- [ ] Add support for gradient clipping
- [ ] Add support for learning rate warm-up
- [ ] Add a timestamp column to `stats.csv`

