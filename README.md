# MetricNet: Recovering Metric Scale in Generative Navigation Policies

[![Project Page](https://img.shields.io/badge/Project%20Page-6cc644&cacheSeconds=60)](https://utn-air.github.io/metricnet/)
[![arXiv](https://img.shields.io/badge/arXiv-2509.13965-b31b1b.svg)](https://arxiv.org/abs/2509.13965)
![GitHub License](https://img.shields.io/github/license/utn-air/metricnet?label=License&color=%23e11d48&cacheSeconds=60)
[![Latest Release](https://img.shields.io/github/v/tag/utn-air/metricnet?label=Latest%20Release&cacheSeconds=60)
](https://github.com/utn-air/metricnet/releases)

## 💡 News

- **January 2026**: Accepted at ICRA 2026! 🎉
- **October 2025**: Training and evaluation code published. Weights and dataset available in releases.

## 🪛 Installation

1) Create a virtual environment with conda and python 3.9. 
```bash
conda create python=3.9 cmake=3.14.0 -n metricnet
conda activate metricnet
```

2) Install the dependencies using pip or pdm

```bash
conda install habitat-sim headless -c conda-forge -c aihabitat
conda env update -n metricnet --file environment.yaml
pip install git+ssh://git@github.com/debOliveira/diffusion_policy.git@db1434cc256b53deb0ad7228c129c0ce7c733822
pip install git+ssh://git@github.com/debOliveira/depth-anything-V2.git@7885bbc0647bc64d55ff5803561ea2c7dea1af72
```

## 🔗 Download data and weights


> [!WARNING]  
> **We are looking into hosting solutions for our data. In the meantime, we make public the data generation scripts.**

1) Download and process the datasets according to [NoMaD's instructions](https://github.com/robodhruv/visualnav-transformer?tab=readme-ov-file#data-wrangling).
2) Generate the benchmark data using the provided script `python generate_benchmark_data.py`. For more information on the generation, please use the `--help` flag.
3) Generate the training data using the provided script `python generate_training_data.py `. For more information on the generation, please use the `--help` flag.
4) Download [DepthAnything-V2 ViT-s weights](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true) for training MetricNet, and [DepthAnything-V2 Metric ViT-B weights](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Base/resolve/main/depth_anything_v2_metric_hypersim_vitb.pth?download=true) for MetricNav.
5) If you want to use the pretrained model, download the weights from [our latest release](https://github.com/utn-air/metricnet/releases).

## 🚀 Training

To train the model, you need to adjust the in the configuration YAML [`metricnet.yaml`](metricnet/config/metric.yaml) as:

1) `train` to `True`
2) `depth_encoder_weights` to the DepthAnythingV2 checkpoint path
3) `datasets/<DATASET>/data_folder`, `datasets/<DATASET>/train` and `datasets/<DATASET>/test` to the folders generated during the (data processing step)[#-download-data-and-weights]

Then, run the following command:

```bash
python train.py -c <YOUR_CONFIG>.yaml
```

If you want to use [wandb](https://wandb.ai/) to log the training, you can set the `use_wandb` flag in the configuration YAML to `True` and  the `project` and `entity` to your desired project and entity (usually your username). Don't forget to login first:
    
```bash
wandb login
```

## 🧪 Testing

To test the model, you need to have the model trained. Weights are available in the [latest release](https://github.com/utn-air/flownav/releases). Adjust in the configuration YAML [`metricnet.yaml`](metricnet/config/metricnet.yaml) as:

1) `train` to `False`.
2) `depth_encoder_weights` to the DepthAnythingV2 checkpoint path.
2) `load_run` to the path of the desired weights.

Then, run the following command:

```bash
python train.py -c <YOUR_CONFIG>.yaml
```

## 📊 Benchmark

To run the benchmark, you need to have the model trained and a local copy of [Matterport3D](https://niessner.github.io/Matterport/). Weights are available in the [latest release](https://github.com/utn-air/flownav/releases). Adjust in the configuration YAML [`benchmark.yaml`](metricnet/config/benchmark.yaml) as:

1) `benchmark_dir` to the path of the Matterport3D dataset.
2) `result_dir` to the desired output folder.
2) `model_cfg` to the path of your model configuration YAML.
2) `model_ckpt` to the path of the desired weights.

Then, run the following command:

```bash
python benchmark.py -c <YOUR_CONFIG>.yaml
```

## 🤖 Deployment

We deployed to a TurtleBot 4 using ROS2 Humble. 

> [!WARNING]  
> **Deployment code is under development and a final version will be uploaded soon.**

## Acknowledgements

- [NoMaD: Goal Masking Diffusion Policies for Navigation and Exploration](https://general-navigation-models.github.io/nomad/index.html)
- [NaviDiffusor: Cost-Guided Diffusion Model for Visual Navigation](https://github.com/SYSU-RoboticsLab/NaviD)
- [Imperative Path Planner (iPlanner)](https://github.com/leggedrobotics/iPlanner)

## 📝 Citation

```
@misc{nayak2025metricnetrecoveringmetricscale,
      title={MetricNet: Recovering Metric Scale in Generative Navigation Policies}, 
      author={Abhijeet Nayak and Débora N. P. Oliveira and Samiran Gode and 
              Cordelia Schmid and Wolfram Burgard},
      year={2025},
      eprint={2509.13965},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.13965} 
  }
```
