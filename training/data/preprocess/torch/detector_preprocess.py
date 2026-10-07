import torch

def detector_preprocessing(inputs):
    # preprocess
    assert len(inputs.shape) == 4 or len(inputs.shape) == 3

    deck_image = torch.from_numpy(inputs).float()
    if len(inputs.shape) == 3:
        deck_image = deck_image.permute([2, 0, 1])
    else:
        deck_image = deck_image.permute([0, 3, 1, 2])

    return deck_image / 255.0