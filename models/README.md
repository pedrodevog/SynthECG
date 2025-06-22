# Models

This directory contains implementations of four state-of-the-art generative deep learning models for ECG signal synthesis, along with supporting components and utilities.

## Model Architectures

### 1. Conditional WaveGAN* ([`cond_wavegan_star.py`](cond_wavegan_star.py))
A modified WaveGAN architecture specifically adapted for ECG generation with conditional batch normalization support.

**Key Components:**
- `CondWaveGANGenerator` - Generator network with transpose convolutions
- `CondWaveGANDiscriminator` - Discriminator with phase shuffle for improved training
- [`ConditionalBatchNorm1d`](cond_batchnorm.py) - Conditional batch normalization for class-aware generation

### 2. Conditional Pulse2Pulse ([`cond_pulse2pulse.py`](cond_pulse2pulse.py))
A U-Net based GAN model designed for pulse-to-pulse ECG generation with conditioning capabilities.

**Key Components:**
- `CondP2PGenerator` - Generator network with transpose convolutions
- `CondP2PDiscriminator` - Discriminator with phase shuffle for improved training
- [`ConditionalBatchNorm1d`](cond_batchnorm.py) - Conditional batch normalization for class-aware generation

### 3. SSSD-ECG ([`SSSD_ECG.py`](SSSD_ECG.py))
Structured State Space Diffusion model that combines S4 layers with diffusion processes for ECG synthesis.

**Key Components:**
- `Residual_block` - Core building block with S4 integration
- `Residual_group` - Groups of residual blocks for hierarchical processing
- S4 integration for long-sequence modeling

### 4. DSAT-ECG ([`DSAT_ECG.py`](DSAT_ECG.py))
Diffusion State Space Augmented Transformer that combines transformer architectures with state space models.

## Supporting Components

### State Space Models (S4)

#### Core S4 Implementation ([`S4Model.py`](S4Model.py))
- `S4` - Main S4 layer implementation
- `SSKernelNPLR` - Normal Plus Low Rank (NPLR) kernel computation
- `HippoSSKernel` - HiPPO-based SSM kernel
- `get_torch_trans` - Transformer encoder utilities

### SPADE Integration

#### SPADE components for improved generation quality ([`SPADEModel.py`](SPADEModel.py))
- `SpadeDecoderLayerBase` - Base decoder layer with attention mechanisms

#### Extended S4 Implementation ([`ELSM.py`](ELSM.py))
Enhanced S4 implementation from the EfficientLongSequenceModeling repository:
- `S4Module` - Modular S4 component for integration
- `SSKernelNPLR` - Optimized NPLR kernel with additional features
- `SSKernelDiag` - Diagonal state matrix variant (S4D)

## Cauchy Kernel Computation
The models support accelerated Cauchy kernel computation through:
- CUDA extensions (recommended for 10-50% speedup)
- PyKeOps backend for GPU acceleration
- Fallback CPU implementation

## Model Configuration

Each model accepts configuration through JSON files located in the [`configs/`](../configs/) directory.

## References

The implementations are based on and adapted from:
- [SSSD-ECG](https://github.com/AI4HealthUOL/SSSD-ECG) - Original SSSD-ECG implementation
- [EfficientLongSequenceModeling](https://github.com/microsoft/EfficientLongSequenceModeling) - SPADE modules and S4 extensions
- [Structured State Spaces](https://github.com/state-spaces/s4) - Original S4 implementation
