import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import matplotlib.pyplot as plt
import librosa
import matplotlib as mpl
import numpy as np
import matplotlib.ticker as mtick

ANNOTATIONS_FILE = './UrbanSound8K/metadata/UrbanSound8K.csv'
AUDIO_DIR = './UrbanSound8K/audio'
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


class UrbanSoundDataset(Dataset):

    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation.to(device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.device = device

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sample_rate = torchaudio.load(audio_sample_path)  # waveform, sample rate

        signal = signal.to(self.device)

        # signal -> Tensor(num_channels, num_samples)

        # resample when the original sample rate != target sample rate
        signal = self._resample_if_necessary(signal, sample_rate)

        # mix down multiple channels to a single channel (mono)
        signal = self._mix_down_if_necessary(signal)

        # if the number of samples is more than the expected, apply cutting operation
        signal = self._cut_if_necessary(signal)

        # if the number of samples is less than the expected, apply right padding operation
        signal = self._right_pad_if_necessary(signal)

        transformed_signal = self.transformation(signal)
        return transformed_signal, label

    def _get_audio_sample_path(self, index):
        fold = f'fold{self.annotations.iloc[index, 5]}'
        filename = self.annotations.iloc[index, 0]
        path = os.path.join(self.audio_dir, fold, filename)
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]

    def _resample_if_necessary(self, signal, sample_rate):
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate
            )
            return resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:  # more than one channel
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_if_necessary(self, signal):
        # signal -> Tensor(num_channels, num_samples) -> (1, ?)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            # e.g. [1, 1, 1] -> [1, 1, 1, 0, 0]
            num_missing_samples = self.num_samples - length_signal

            """
            pad: (x, y), x: number of item to prepend, y: number of item to append
            if pad = (a, b, c, d):
            a: prepend at last dimension; b: append at last dimension
            c: prepend at second last dimension; d: append at second last dimension
            """
            last_dim_padding = (0, num_missing_samples)  # x = 0, y = num_missing_samples

            signal = torch.nn.functional.pad(input=signal, pad=last_dim_padding)
        return signal


def plot_spectrogram(spectrogram):
    # https://stackoverflow.com/questions/46031397/using-librosa-to-plot-a-mel-spectrogram

    mpl.use('TkAgg')  # FIXME: !IMPORTANT

    plt.figure(figsize=(6, 4))
    # plt.xticks(fontsize=5)

    # Convert a power spectrogram (amplitude squared) to decibel (dB) units
    dB = librosa.power_to_db(spectrogram[0], ref=np.max)
    librosa.display.specshow(dB, y_axis='mel', fmax=8000, x_axis='time')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))

    plt.colorbar(format='%2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()


def check_device():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using device {device}.')
    return device


if __name__ == '__main__':
    device = check_device()

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    urban_sound_dataset = UrbanSoundDataset(
        annotations_file=ANNOTATIONS_FILE,
        audio_dir=AUDIO_DIR,
        transformation=mel_spectrogram,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        device=device
    )
    print(f'There are {len(urban_sound_dataset)} samples in the dataset.')

    print('\nThe first audio file (after resampling and being mixed down):')
    mel_spectrogram, label = urban_sound_dataset[0]
    print(f'Mel_spectrogram (signal): {mel_spectrogram}')
    print(f'Shape of the signal: {mel_spectrogram.shape}')
    print(f'Label of the signal: {label}\n')

    plot_spectrogram(mel_spectrogram)
