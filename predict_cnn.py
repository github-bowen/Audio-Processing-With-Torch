from torch import Tensor

from cnn import CNNNetwork
from urban_sound_dataset import *

class_mapping = [  # index to class(target)
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]


def predict(model, inputs, targets, class_mapping):
    model.eval()  # for evaluation;      model.train() for training
    with torch.no_grad():  # no gradient
        predictions = model(inputs)  # in out case: Tensor(1, 10): 1 sample, 10 class(features)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[targets]
    return predicted, expected


if __name__ == '__main__':
    # load back model
    cnn = CNNNetwork()
    state_dict = torch.load('cnn.pth', map_location=torch.device('cpu'))
    cnn.load_state_dict(state_dict)

    # load urban sound dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    device = check_device()

    urban_sound_dataset = UrbanSoundDataset(
        annotations_file=ANNOTATIONS_FILE,
        audio_dir=AUDIO_DIR,
        transformation=mel_spectrogram,
        target_sample_rate=SAMPLE_RATE,
        num_samples=NUM_SAMPLES,
        device=device
    )

    inputs, targets = urban_sound_dataset[0][0], urban_sound_dataset[0][1]
    inputs.unsqueeze_(0)

    # make an inference
    predicted, expected = predict(cnn, inputs, targets, class_mapping)

    print(f'Predicted: {predicted}')
    print(f'Expected: {expected}')
