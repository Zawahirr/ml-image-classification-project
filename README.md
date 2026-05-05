# ML Image Classification Project

## Overview
This project focuses on image classification using deep learning models, including MLPs and CNNs. The goal is to compare multiple architectures across different datasets and analyze their performance.

## Planned Datasets
- Fashion-MNIST
- CIFAR-10

## Planned Models
- MLP
- Simple CNN
- LeNet
- Improved Custom CNN

## Structure
- `src/` -> model + training code
- `notebooks/` -> exploration + visualization
- `outputs/` -> results + figures
- `reports/` -> report drafts
- `notes/` -> learning + planning notes

## First Experiment

Train the Fashion-MNIST MLP baseline:

```powershell
python -m src.train --dataset fashion_mnist --model mlp --epochs 5
```

Train the Fashion-MNIST simple CNN:

```powershell
python -m src.train --dataset fashion_mnist --model simple_cnn --epochs 5
```

Train the Fashion-MNIST LeNet baseline:

```powershell
python -m src.train --dataset fashion_mnist --model lenet --epochs 5
```

Train the Fashion-MNIST improved custom CNN:

```powershell
python -m src.train --dataset fashion_mnist --model custom_cnn --epochs 5
```

Each run saves a config file, training history, learning curves, classification report, confusion matrix, and summary metrics under `outputs/`.

The same commands can be repeated with `--dataset cifar10` after the Fashion-MNIST pipeline is validated.

Create a report-ready comparison table after running experiments:

```powershell
python -m src.summarize_results
```

## Status
Fashion-MNIST train/validation/test pipeline initialized with MLP, Simple CNN, LeNet, and custom CNN model options.
