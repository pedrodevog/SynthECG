# SynthECG 

![SynthECG](https://img.shields.io/badge/SynthECG-Ready-brightgreen) ![Python](https://img.shields.io/badge/Python-3.11%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-orange)

This project contains all the code to perform a systematic evaluation of a model able to produce conditional 10-sec 12-lead ECGs.

## Models

The code includes four state-of-the-art generative deep learning models to generate the ECG signals:

- **WaveGAN\*** - Modified WaveGAN architecture for ECG generation
- **Pulse2Pulse** - U-Net based GAN model
- **SSSD-ECG** - Structured State Space Diffusion model
- **DSAT-ECG** - Diffusion State Space Augmented Transformer

If you want to add your model, just fork this repo and set up a pull request. I will try to review the code as soon as possible.

## Setup

### Requirements
- **GPU**: NVIDIA GPU with CUDA capability (minimum 16GB VRAM, recommended 24GB+)
- **CUDA**: 12.4.0
- **GCC**: 11.3.0
- **Python**: 3.11

### Linux

Setting up the codebase in a linux environment should be pretty straightforward:

1. Clone the repository
2. Create and activate a virtual environment with Python3.11
3. Install dependencies: `pip install -r requirements.txt`
4. Build CUDA extensions: `python setup.py install`

Do not forget to prepare the PTB-XL data and update the config files for the models.
You should now be ready to go!

### Windows/Docker Setup

Setting up on Windows or other non-Linux systems requires Docker with GPU support. 

**📋 [Docker Setup Guide](.devcontainer/README.md)**

After setting up Docker, follow the Linux setup steps above.


## Features

### Training

`train_wavegan.py`

`train_p2p.py`

`train_sssd.py`

`train_dsat.py`

Train each of the models.

### Sampling

`sample.py`

Sample trained model checkpoints matching PTB-XL dataset.

### Evaluating

`evaluate.py`

For both validation and tesing.

| Metric | Explanation |
|--------|-------------|
| **MMD** | Maximum Mean Discrepancy - measures distributional differences between real and synthetic data |
| **PSD-MMD** | Power Spectral Density MMD - MMD applied to frequency domain representations |
| **PSD-PRD** | Power Spectral Density Percent Root-mean-square Difference - PRD applied to frequency domain representations |
| **FID<sub>ECG</sub>** | Fréchet Inception Distance for ECG - measures quality and diversity of generated ECG signals |
| **KID<sub>ECG</sub>** | Kernel Inception Distance for ECG - alternative to FID using kernel methods |
| **TSTR** | Train on Synthetic, Test on Real - evaluates utility of synthetic data for downstream tasks |
| **TRTS** | Train on Real, Test on Synthetic - measures how well synthetic data represents real data patterns |
| **NMI** | Normalized Mutual Information - measures statistical dependence between leads |

#### Also available:

- RMSE
- PRD
- DTW
- discrete FD

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{devogelaere2025synthecg,
  title={A Systematic Evaluation Framework of Generative Deep Learning for 10-second 12-lead Synthetic ECG Signals},
  author={Devogelaere, Pedro and Van Santvliet, Lore and De Vos, Maarten},
  year={2025},
  school={KU Leuven}
}
```
