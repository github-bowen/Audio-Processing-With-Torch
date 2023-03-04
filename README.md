# Codes and notes for learning audio operations with pytorch

## Repo Info

- Related Series in YouTube: [PyTorch for Audio + Music Processing](https://www.youtube.com/watch?v=gp2wZqDoJ1Y&t=2s)

  Codes given by the author: [GitHub Repo](https://github.com/musikalkemist/pytorchforaudio)

- Files / Directories Info (Listed in the order explained in the tutorial above):

  Directories:

  - `MNIST`: A dataset, auto downloaded in file `train.py`.
  - `UrbanSound8K`: A dataset, downloaded from website [URBANSOUND8K DATASET](https://urbansounddataset.weebly.com/urbansound8k.html)

  Files:

  - `train.py`: Contains a class `FeedForwardNet` and functions `download_mnist_datasets`, `train_one_epoch` and `train` which are used for downloading MNIST dataset and training them using FeedForwardNet model.
  - `feedforwardnet.pth`: Model saved from `train.py`.
  - `inference.py`: Contains a function `predict` for validating the model `feedforwardnet.pth`.
  - `urban_sound_dataset.py`: Contains a class `UrbanSoundDataset` for loading `.wav` sound file in urbansound8k dataset and get the waveform signals and sample rates of each audio.

## Some Environments Problems

- Get `RuntimeError: No audio I/O backend is available.` message while running code `torchaudio.load(audio_sample_path)` at file `urban_sound_dataset.py`:

  ```shell
  # try with commands below
  pip install SoundFile
  # or
  pip install sox
  ```

  