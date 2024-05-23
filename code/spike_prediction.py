import torch
import numpy as np
from typing import List
from torch.nn import Module
from model import NeuralNetwork_2c
from dataset import one_hot_encode, one_hot_decode

PEPTIDE_LENGTH = 9


def extract_spike_data():
    with open('../spike.txt', 'r') as input_file:
        spike_input = input_file.read().strip().replace('\n', '')
        while len(spike_input) >= 9:
            yield one_hot_encode(spike_input[:9]).float()
            spike_input = spike_input[1:]


def predict_top_n(model: Module, peptides: List[torch.Tensor], n: int):
    predictions = np.zeros(len(peptides))
    for idx, peptide in enumerate(peptides):
        predictions[idx] = model.forward(peptide)
    ind = np.argpartition(predictions, -n)[-n:][::-1]
    return [(one_hot_decode(peptides[i]), predictions[i]) for i in ind]


def main():
    model = NeuralNetwork_2c()
    model.load_state_dict(torch.load('../models/2c_model.pth'))

    peptides = list(extract_spike_data())
    best_detectable_3 = predict_top_n(model, peptides, 3)
    print(best_detectable_3)


if __name__ == '__main__':
    main()
