# νGAN: A Deep Learning Emulator for Cosmic Web Simulations with Massive Neutrinos

![Generated cosmological maps](plots/processed_for_paper/sample_maps2.png)

This is a **PyTorch implementation of νGAN, a conditional Wasserstein GAN (WGAN-GP)** designed to generate 2D cosmological maps conditioned on neutrino masses. The training pipeline supports spectral loss regularization, gradient penalty, multi-GPU training, and flexible optimization schedules.

**Paper** is here: https://iopscience.iop.org/article/10.3847/1538-4357/ae3de4.

**Data** is here: https://zenodo.org/records/18224158

**Interactive website** for using νGAN for inference is here: https://kaushallab.mtu.edu/nugan. We welcome any new features you'd like to have.

![Generated cosmological maps](plots/processed_for_paper/sample_maps.png)




---

## Overview and Results

The main training script (`train.py`) trains a neutrino-conditioned GAN that:
- Learns from 2D cosmological density maps
- Conditions generation on neutrino mass parameters
- Uses Wasserstein loss with gradient penalty
- Incorporates a power-spectrum–based spectral loss
- Saves best models based on multiple criteria (G loss, G+D loss, power spectrum loss)
<br><br>

<p align="center">
  <img src="plots/processed_for_paper/power_spectra.png" width="1500">
  <br>
  <em>Mean power spectrum for 1000 samples</em>
</p>

<br><br>

<p align="center">
  <img src="plots/processed_for_paper/pixel_intensity.png" width="1500">
  <br>
  <em>Mean pixel intensity histogram for 1000 samples</em>
</p>

---

## Repository Structure

```text
.
├── train.py        # Main training script
├── utils.py        # Helper utilities
├── models.py       # Generator and Critic architectures
├── run.sh          # Example shell script for running training script
├── infer.ipynb     # Example notebook for inference
├── env.yml         # Conda environment file
├── README.md

```

---

## Data

The data and trained model are hosted at https://zenodo.org/records/18224158. The data includes processed 2D density maps and corresponding neutrino mass labels used for training. Note that the training maps are normalized to [-1,1]. The model checkpoint (pytorch state_dict) corresponds to trained νGAN used in the analysis presented in the associated manuscript (See the paper for more details). Checkpoints and data should be moved to `/nugan/checkpoints/` and `/nugan/data/` to be used for inference and training. The data and model are provided without restrictions and may be used for research and educational purposes.  Note that the neutrino mass and corresponding maps in train set range from [0:3000] for 0.0 eV, [3000:6000] for 0.1 eV, [6000:9000] for 0.4 eV, [9000:12000] for 0.8 eV, and [12000:] for 1.2 eV because each realization gives 1500 maps (500 along each axis) and we use 2 realizations for training. For test data, we only use 1 realization, so the effective indices become [0:1500], [1500:3000], [3000:4500], [4500:6000], and [6000:7500] respectively for 0.0, 0.1, 0.4, 0.8, and 1.2 eV.

<p align="center">
  <img src="plots/processed_for_paper/data_distribution.png" width="500">

  <br>
  <em>Pixel intensity distribution for 15,000 training data maps of shape (256,256). Note the high variance at the right tail.</em>
</p>


## Requirements

The recommended way to set up the environment is using the provided Conda environment file.
```bash
conda env create -f env.yml
conda activate nugan
```

## Run

Modify train params inside run.sh. Run as `bash run.sh`

During training, the following directories are created inside save_path:
```text

save_path/
├── saved_models/
│   ├── best_model_on_G_loss.pt
│   ├── best_model_on_GD_loss.pt
│   ├── best_model_on_pk_loss.pt
│   └── model_epoch_*.pt
├── saved_losses/
│   ├── G_losses.csv
│   ├── D_losses.csv
│   ├── D_losses_real.csv
│   ├── D_losses_fake.csv
│   └── pk_losses.csv
├── saved_data/
│   └── img_list.npy
└── args.txt
```

## Improvements

- Critic contains `BatchNorm` unlike standard WGAN practice.
- Critic is updated using Wasserstein loss with gradient penalty and spectral loss.

- Critic gradients are clipped

- Spectral loss is enabled after an initial warm-up period (default: 5000 iterations).

- Generator updates can be controlled via:
    - `G_update_interval` is the number of times the Critic is updated for each update in Generator.
    - `G_update_freq` is the number of times the Generator is updated for each update in Critic.
    - One of the above must be `0` when the other is not.

- Neutrino masses are used to condition both the Generator and the Critic. They are concatenated `mchn` times at the start and the end of the 1-D latent vector `nz`.

- Fixed noise and conditioning vectors are used for periodic sample generation.

- Checkpoints are saved based on lowest running losses on Generator, Generator+Critic, and the average binned P(k) loss between real and fake data (invere-scaled) batches of `G_samples` samples.
