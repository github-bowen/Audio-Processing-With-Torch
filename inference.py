import torch
from torch import Tensor
from train import FeedForwardNet, download_mnist_datasets

class_mapping = [i for i in range(10)]  # index to class(target)


def predict(model, inputs, targets, class_mapping):
    model.eval()  # for evaluation;      model.train() for training
    with torch.no_grad():  # no gradient
        predicted, expected = [], []
        for i in range(len(inputs)):
            predictions = model(inputs[i])  # in out case: Tensor(1, 10): 1 sample, 10 class(features)
            predicted_index = predictions[0].argmax(0)
            predicted.append(class_mapping[predicted_index])
            expected.append(class_mapping[targets[i]])
    return predicted, expected


if __name__ == '__main__':
    # load back model
    feed_forward_net = FeedForwardNet()
    state_dict = torch.load('feedforwardnet.pth')
    feed_forward_net.load_state_dict(state_dict)

    # load MNIST validation dataset
    _, validation_data = download_mnist_datasets()

    # get all sample from the validation dataset for inference
    inputs, targets = [], []
    count = 40  # first 40 samples
    for (input, target) in validation_data:
        count -= 1
        inputs.append(input)
        targets.append(target)
        if count <= 0:
            break
    # inputs, targets = validation_data[0][0], validation_data[0][1]  # validation[0][x] the first sample

    # make an inference
    predicted, expected = predict(feed_forward_net, inputs, targets, class_mapping)

    print(f'Predicted: {predicted}')
    print(f'Expected: {expected}')
    print(Tensor(predicted) == Tensor(expected))
