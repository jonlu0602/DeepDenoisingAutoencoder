# Deep Denoising Autoencoder (DDAE) for Speech Enhancement


Tensorflow implementation of [Speech Enhancement Based on Deep Denoising Autoencoder](https://www.isca-speech.org/archive/archive_papers/interspeech_2013/i13_0436.pdf)

## Getting Started

<!-- These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system. -->

Clone This repository to your local machine.

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


```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Yu-Ding Lu** - *DDAE* - [DDAE](https://github.com/jonlu0602)



## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
