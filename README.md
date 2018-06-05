# Deep Denoising Autoencoder (DDAE) for Speech Enhancement


Tensorflow implementation of [Speech Enhancement Based on Deep Denoising Autoencoder](https://www.isca-speech.org/archive/archive_papers/interspeech_2013/i13_0436.pdf)

## Getting Started

<!-- These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system. -->

Clone This repository to your local machine and run python/create_dir.sh first.

### Prerequisites

* python                 3.5
* tensorflow-gpu         1.8.0
* scikit-learn           0.19.1
* scipy                  1.1.0
* h5py                   2.7.1
* librosa                0.5.1
* numpy                  1.14.3
* tqdm                   4.23.2

### Getting Started

1. Download free dataset from [VoxForge](http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Original/16kHz_16bit/) for clean data. Here I would recommed download cmu_us_awb_arctic.tgz
2. Unzip clean dataset to /DeepDenoisingAutoencoder/data/raw/clean/
3. Download free dataset from [ESC-50](https://github.com/karoldvl/ESC-50) for noise data. 
4. Move ESC-50-master/audio to /DeepDenoisingAutoencoder/data/raw/noise/
5. Set parameters in python/main.py
6. Run python/main.py

## Result

![Spectrogram on Test data](https://github.com/jonlu0602/DeepDenoisingAutoencoder/blob/master/pic1.png)

## Deployment

You can read many comments inside all .py files.

## Authors

**Yu-Ding Lu** - [Linkedin](https://www.linkedin.com/in/yu-ding-lu-40231b139/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Bio-ASP lab - CITI - Academia Sinica
