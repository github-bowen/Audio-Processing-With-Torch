import torch
from torch import nn
from torch.utils.data import DataLoader
import torchaudio

from urban_sound_dataset import UrbanSoundDataset, ANNOTATIONS_FILE, \
    AUDIO_DIR, SAMPLE_RATE, NUM_SAMPLES, check_device
from cnn import CNNNetwork

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = .001


def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # back propagate loss and update weights
        optimizer.zero_grad()  # reset the gradient to zero
        loss.backward()
        optimizer.step()

    print(f'Loss: {loss.item()}')


def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f'Epoch {i + 1}')
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print('-----------------------------------')
    print('Training is done')


if __name__ == '__main__':

    device = check_device()

    # instantiate dataset object: urban sound dataset
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

    # create a dataloader for the train set
    train_data_loader = DataLoader(urban_sound_dataset, batch_size=BATCH_SIZE)

    # build model
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using {device} device')
    cnn = CNNNetwork().to(device)

    # instantiate loss function + optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=cnn.parameters(),
        lr=LEARNING_RATE
    )

    # train model
    train(cnn, train_data_loader, loss_fn, optimizer, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), 'cnn.pth')
    print('Model trained and stored at cnn.pth')
