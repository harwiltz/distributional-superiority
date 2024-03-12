# Distributional Superiority

This repository contains the reference implementation of the \[DAU+\]DSUP($q$) algorithms presented in:

**[Action Gaps and Advantages in Continuous-Time Distributional Reinforcement Learning](https://arxiv.org/abs/2410.11022)**

by **[Harley Wiltzer](https://harwiltz.github.io)**\*, [Marc G. Bellemare](http://www.marcgbellemare.info), [David Meger](https://www.cim.mcgill.ca/~dmeger), [Patrick Shafto](https://patrickshafto.com), and **[Yash Jhaveri](https://yash-jhaveri.github.io)**\*.

## Setup

This project uses [PDM](https://pdm-project.org/latest/) for dependency management. See <https://pdm-project.org/latest/#installation> for installation instructions.

Once PDM has been installed, execute the following from the project root to sync the dependencies:

```bash
pdm venv create
pdm install
```

Before running any code, be sure to activate the virtual environment (from the project root):

```bash
source .venv/bin/activate
```

## Downloading Data
Some environments simulate dynamics from datasets. The `download_data.sh` file downloads these datasets. Make this
script executable:

```bash
chmod +x download_data.sh
```

Then run the script to download the datasets:

```bash
./download_data.sh
```

This script will create a `data` directory in the project root with the requisite datasets.

## Training an Agent

The easiest way to run training scripts is with our `justfile`, using the [`just`](https://just.systems/) command runner.

### Risk Neutral Simulation
To train agents for risk-neutral option trading, execute

```bash
just writer=[aim | comet] agent=[dsup | qrdqn | dau] option_idx=<int> time_mul=<int> train_options
```

Here, `option_idx` specifies the commodity for the environment, and `time_mul` is the decision frequency. Setting `time_mul=1` results in the base frequency, and `time_mul=n` is `n` times the base frequency.

To train the DAU+DSUP(1/2) variant, execute replace `train_options` with `train_options_dsup_shifted`.

### CVaR Simulation
To train agents for risk-sensitive option trading with CVaR, execute

```bash
just writer=[aim | comet] agent=[dsup | qrdqn | dau] option_idx=<int> time_mul=<int> risk_param=<float> train_options_risky
```

Here, `risk_param` refers to the CVaR level for the experiment.

## Citation

If you build on our work or find it useful, please cite it using the following bibtex,

```bibtex
@inproceedings{wiltzer2024action,
  title={Action Gaps and Advantages in Continuous-Time Distributional Reinforcement Learning},
  author={Harley Wiltzer and Marc G. Bellemare and David Meger and Patrick Shafto and Yash Jhaveri},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=BRW0MKJ7Rr}
}
```
