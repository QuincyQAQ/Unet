# U-Net Image Restoration Training Script

## Overview

This repository contains a Python script for training a U-Net model on image restoration tasks. The script utilizes PyTorch, Accelerate, and various metrics to train and evaluate the model. The training process includes data loading, model training, validation, and metric logging.

## Features

- **Model Training**: Utilizes a U-Net architecture for image restoration.
- **Data Loading**: Loads training and validation data using PyTorch's `DataLoader`.
- **Metrics**: Computes and logs metrics such as PSNR, SSIM, MAE, and LPIPS.
- **Acceleration**: Uses the `Accelerate` library for distributed training and mixed-precision.
- **Logging**: Logs training metrics to a CSV file and plots them using Matplotlib.
- **Checkpointing**: Saves the best model based on PSNR.

## Requirements

- Python 3.7+
- PyTorch
- TorchMetrics
- PyTorch-SSIM
- Accelerate
- Matplotlib
- CSV

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/unet-image-restoration.git
   cd unet-image-restoration
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The training configuration is managed via a YAML file (`config.yml`). The configuration includes parameters for data directories, model settings, optimization settings, and training hyperparameters.

## Usage

To start training, run the `train.py` script:

```bash
python train.py
```

## Training Process

1. **Data Loading**: The script loads training and validation data using the `get_training_data` and `get_validation_data` functions from the `data` module.
2. **Model Initialization**: The U-Net model is initialized and moved to the appropriate device (CPU or GPU).
3. **Loss Functions**: The script uses PSNR, SSIM, and LPIPS as loss functions.
4. **Optimization**: AdamW optimizer with a cosine annealing learning rate scheduler is used for optimization.
5. **Training Loop**: The model is trained for a specified number of epochs. After each epoch, the model is validated, and metrics are logged.
6. **Checkpointing**: The best model based on PSNR is saved.
7. **Plotting**: After training, the metrics are plotted and saved as an image.

## Metrics

The following metrics are computed and logged during training:

- **PSNR (Peak Signal-to-Noise Ratio)**
- **SSIM (Structural Similarity Index Measure)**
- **MAE (Mean Absolute Error)**
- **LPIPS (Learned Perceptual Image Patch Similarity)**

## Results

The training results, including metrics and plots, are saved in the `runs/exp` directory. Each training run creates a new subdirectory with a unique number.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

For any questions or issues, please open an issue on GitHub.