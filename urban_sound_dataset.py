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


class UrbanSoundDataset(Dataset):

    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sample_rate = torchaudio.load(audio_sample_path)  # waveform, sample rate

        # signal -> Tensor(num_channels, samples)

        # resample when the original sample rate != target sample rate
        signal = self._resample_if_necessary(signal, sample_rate)

        # mix down multiple channels to a single channel (mono)
        signal = self._mix_down_if_necessary(signal)

        transformed_signal = self.transformation(signal)
        return signal, label, transformed_signal

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


def plot_spectrogram(spectrogram):
    # https://stackoverflow.com/questions/46031397/using-librosa-to-plot-a-mel-spectrogram

    mpl.use('TkAgg')  # FIXME: !IMPORTANT

    plt.figure(figsize=(6, 4))
    # plt.xticks(fontsize=5)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))

    # Convert a power spectrogram (amplitude squared) to decibel (dB) units
    dB = librosa.power_to_db(spectrogram[0], ref=np.max)
    librosa.display.specshow(dB, y_axis='mel', fmax=8000, x_axis='time')

    plt.colorbar(format='%2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    ANNOTATIONS_FILE = './UrbanSound8K/metadata/UrbanSound8K.csv'
    AUDIO_DIR = './UrbanSound8K/audio'
    SAMPLE_RATE = 64000

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        n_mels=64
    )

    urban_sound_dataset = UrbanSoundDataset(
        annotations_file=ANNOTATIONS_FILE,
        audio_dir=AUDIO_DIR,
        transformation=mel_spectrogram,
        target_sample_rate=SAMPLE_RATE
    )
    print(f'There are {len(urban_sound_dataset)} samples in the dataset.')

    print('\nThe first audio file (after resampling and being mixed down):')
    signal, label, mel_spectrogram = urban_sound_dataset[0]
    print(f'Signal: {signal}')
    print(f'Shape of the signal: {signal.shape}')
    print(f'Label of the signal: {label}\n')

    plot_spectrogram(mel_spectrogram)
