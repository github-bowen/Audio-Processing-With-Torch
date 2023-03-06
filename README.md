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
  - `urban_sound_dataset.py`: Contains a class `UrbanSoundDataset` for loading `.wav` sound file in urbansound8k dataset and getting the waveform signals, sample rates and mel-spectorgrams of each audio. Serveral works are done in method `__getitem__`:
    - Load the `.wav` audio file and get its waveform signal and sample rate.
    - Resample the signal if the original sample rate is not equal to the target sample rate.
    - Mix down multiple channels to moto.
    - If the number of samples is more than the expected, apply cutting operation.
    - if the number of samples is less than the expected, apply right padding operation.
    - Use transforming function (here it's `mel_spectrogram`) to transform it.

## Some Environmental Problems

- Get `RuntimeError: No audio I/O backend is available.` message while running code `torchaudio.load(audio_sample_path)` at file `urban_sound_dataset.py`:

  ```shell
  # try with commands below
  pip install SoundFile
  # or
  pip install sox
  ```

- Get error message below when plotting mel-spectrogram using matplotlib:

  ```none
   manager_pyplot_show = vars(manager_class).get("pyplot_show")
  TypeError: vars() argument must have __dict__ attribute
  ```

  Solutions ([Stack Overflow](https://stackoverflow.com/questions/75453995/pandas-plot-vars-argument-must-have-dict-attribute)):

  ```python
  mpl.use('TkAgg')  # Add this code
  ```

  