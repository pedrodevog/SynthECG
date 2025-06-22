# SynthECG: First Standardized Evaluation Framework for Synthetic ECGs

![SynthECG](https://img.shields.io/badge/SynthECG-Ready-brightgreen) ![Python](https://img.shields.io/badge/Python-3.11%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-orange)

This project contains all the code to perform a systematic evaluation of a model able to produce conditional 10-sec 12-lead ECGs.

| PTB-XL Dataset | DSAT-ECG Generated |
|----------------|-------------------|
| ![PTB-XL NORM Sample](images/PTBXL_norm_with_stats.jpg) | ![DSAT-ECG NORM Sample](images/DSAT_norm_with_stats.jpg) |

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


### Training & Evaluation Pipeline

The SynthECG framework provides a complete pipeline for training generative models and evaluating their performance on ECG synthesis tasks.

| Phase | Script | Description |
|-------|--------|-------------|
| **Training** | [`train_wavegan.py`](train_wavegan.py) | Train the WaveGAN* model on ECG data |
| | [`train_p2p.py`](train_p2p.py) | Train the Pulse2Pulse U-Net GAN model |
| | [`train_sssd.py`](train_sssd.py) | Train the SSSD-ECG diffusion model |
| | [`train_dsat.py`](train_dsat.py) | Train the DSAT-ECG transformer model |
| **Sampling** | [`sample.py`](sample.py) | Generate synthetic ECG signals from trained models |
| **Evaluating** | [`evaluate.py`](evaluate.py) | Comprehensive evaluation of synthetic vs. real ECG quality |

### Core Evaluation Metrics

Our framework provides comprehensive evaluation across multiple dimensions of ECG signal quality:

| Metric | Description |
|--------|-------------|
| **MMD** | Maximum Mean Discrepancy - measures statistical differences between real and synthetic ECG distributions |
| **PSD-MMD** | Power Spectral Density MMD - evaluates frequency domain characteristics |
| **PSD-PRD** | Power Spectral Density Percent Root-mean-square Difference - frequency domain signal fidelity |
| **FID<sub>ECG</sub>** | Fréchet Inception Distance for ECG - measures realism and diversity of generated signals |
| **KID<sub>ECG</sub>** | Kernel Inception Distance for ECG - robust alternative to FID using kernel embeddings |
| **TSTR** | Train on Synthetic, Test on Real - evaluates downstream task performance using synthetic data |
| **TRTS** | Train on Real, Test on Synthetic - measures representativeness of synthetic data |
| **NMI** | Normalized Mutual Information - assesses physiological relationships between ECG leads |

### Additional Metrics

> **Note**: The following metrics are available in the framework but were not included in the final evaluation due to their limitations for ECG signal assessment. They are primarily included for research completeness and comparison with existing literature.

| Metric | Description | Note |
|--------|-------------|------|
| **RMSE** | Root Mean Square Error for point-wise signal comparison | Limited utility for time-series with natural variability |
| **PRD** | Percent Root-mean-square Difference for point-wise signal comparison | Sensitive to minor temporal shifts |
| **DTW** | Dynamic Time Warping for temporal alignment assessment | Computationally expensive for large datasets |
| **Discrete FD** | Discrete Fréchet distance for temporal alignment assessment | May not capture physiological ECG characteristics |

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
