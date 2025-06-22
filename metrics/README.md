# Multi-Label Balanced Metric Evaluation

A robust implementation for evaluating metrics on multi-label datasets, with an emphasis on class-balanced batch processing.

## Overview

This folder contains an implementation for running evaluation metrics on multi-label datasets while ensuring balanced representation of all classes. The core function `run_metric` in the Metric.py file processes input data in batches, maintaining class balance even when samples belong to multiple classes simultaneously.

## Problem Statement

When evaluating metrics on multi-label datasets, traditional batch sampling techniques often lead to unbalanced class representation because:

1. Samples belong to multiple classes simultaneously
2. Some classes may be "starved" when their samples are allocated to other classes first
3. Without tracking sample usage across batches, some samples may be overrepresented

## Features

- **Balanced batch sampling** for multi-label classification datasets
- **Sample usage tracking** to ensure uniform coverage of the dataset
- **Weighted selection** prioritizing underrepresented samples and classes
- **Per-class metric reporting** with mean and standard deviation
- **Batch processing** to handle large datasets efficiently

## Core Algorithm

The balanced batch construction algorithm works in these key steps:

1. **Initialization**:
   - Track sample usage counts across batches
   - Calculate target samples per class based on overall distribution

2. **Two-Phase Batch Construction**:
   - **Phase 1**: Satisfy minimum requirements for each class using weighted sampling
   - **Phase 2**: Fill remaining batch slots with weighted random sampling

3. **Weighted Selection**:
   - Uses inverse frequency weighting so less-used samples are more likely to be selected
   - Implements weighted sampling without replacement

4. **Metric Calculation**:
   - Computes metrics overall and per-class
   - Aggregates statistics across batches

## Extending

To use a custom metric function, simply add a Python file and define a function that takes two dataset inputs and returns a scalar value:

```python
def custom_metric(X, Y):
    """
    Custom metric function.
    
    Parameters:
    -----------
    X : torch.Tensor
        First tensor.
    Y : torch.Tensor
        Second tensor.
        
    Returns:
    --------
    float
        Metric value.
    """
    # Your metric calculation
    return result
```

Then pass this function to `run_metric`.
