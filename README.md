# CPL on Real Human Data.

To test Contrastive Preference Learning (CPL) on real human data, we use this fork of the [Direct Preference-based Policy Optimization without Reward Modeling](https://arxiv.org/abs/2301.12842) codebase (found [here]()), which is also a fork of [PreferenceTransformer](https://github.com/csmile-1006/PreferenceTransformer), which was based on [FlaxModels](https://github.com/matthias-wright/flaxmodels) and [IQL](https://github.com/ikostrikov/implicit_q_learning).

To move from DPPO to CPL, we just change the loss function and move to a probabilistic policy. We leave all hyperparameters of the preference model the same.


# Installation 

Note: Our code was tested on Linux OS with CUDA 12. If your system specification differs (e.g., CUDA 11), you may need to modify the `requirements.txt` file and the installation commands.

To install the required packages using Conda, follow the steps below:
```
conda create -n dppo python=3.8
conda activate dppo

conda install -c "nvidia/label/cuda-12.3.0" cuda-nvcc
pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

# How to run CPL

Note: we use the same hyperparameters for the preference model (NU and M) from DPPO.
## Train preference model

```
python -m JaxPref.new_preference_reward_main --env_name [ENV_NAME] --seed [SEED] --transformer.smooth_w [NU] --smooth_sigma [M] 
```

## Train agent

```
XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1 python train.py --env_name [ENV_NAME] --seed [SEED] --transformer.smooth_w [NU] --smooth_sigma [M] --dropout 0.25 --cpl True --lambd 0.7 --dist_temperature 0.1
```
