import torch
from torch.utils.data import Dataset
from typing import List, Dict
import numpy as np

AMINO_ACIDS_TYPES = 'SCYIGTRMAHPEVFLNKWDQ'
amino_acids_dict = {amino: idx for idx, amino in enumerate(AMINO_ACIDS_TYPES)}


def peptide_to_indices(peptide: str) -> torch.Tensor:
    return torch.tensor(
        [amino_acids_dict[ch] for ch in peptide]
    )


def one_hot_encode(peptide: str) -> torch.Tensor:
    return torch.nn.functional.one_hot(
        peptide_to_indices(peptide),
        len(amino_acids_dict)
    )


def one_hot_decode(one_hot: torch.Tensor) -> str:
    return ''.join([AMINO_ACIDS_TYPES[np.argmax(vec)] for vec in one_hot])


class PeptidesDataset(Dataset):
    peptides: List[str]
    labels: List[int]
    amino_acids: Dict[chr, int]

    def __init__(self, positive_samples_filepath, negative_samples_file_path):
        with open(positive_samples_filepath) as positive_dataset:
            positive_peptides = positive_dataset.read().strip().split('\n')
            positive_peptides = 9 * positive_peptides

        with open(negative_samples_file_path) as negative_dataset:
            negative_peptides = negative_dataset.read().strip().split('\n')

        self.peptides = positive_peptides + negative_peptides
        self.labels = torch.tensor([1.0] * len(positive_peptides) + [0.0] * len(negative_peptides), dtype=torch.float)

    def __len__(self) -> int:
        return len(self.peptides)

    def __getitem__(self, index):
        x = one_hot_encode(self.peptides[index]).float()
        y = self.labels[index]
        return x, y
