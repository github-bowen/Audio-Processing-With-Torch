import os
from torch.utils.data import Dataset
import pandas as pd
import torchaudio


class UrbanSoundDataset(Dataset):

    def __init__(self, annotations_file, audio_dir):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sample_rate = torchaudio.load(audio_sample_path)  # waveform, sample rate
        return signal, label

    def _get_audio_sample_path(self, index):
        fold = f'fold{self.annotations.iloc[index, 5]}'
        filename = self.annotations.iloc[index, 0]
        path = os.path.join(self.audio_dir, fold, filename)
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]


if __name__ == '__main__':
    ANNOTATIONS_FILE = './UrbanSound8K/metadata/UrbanSound8K.csv'
    AUDIO_DIR = './UrbanSound8K/audio'

    urban_sound_dataset = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR)

    print(f'There are {len(urban_sound_dataset)} samples in the dataset.')

    print('The first audio file:')
    signal, label = urban_sound_dataset[0]
    print(f'Signal: {signal}\nShape of the signal: {signal.shape}\nLabel of the signal: {label}')
