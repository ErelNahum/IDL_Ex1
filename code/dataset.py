import torch
from torch.utils.data import Dataset
from typing import List, Dict
import numpy as np

AMINO_ACIDS_TYPES = 'SCYIGTRMAHPEVFLNKWDQ'

class PeptidesDataset(Dataset):
    peptides: List[str]
    labels: List[int]
    amino_acids: Dict[chr, int]

    def __init__(self, positive_samples_filepath, negative_samples_file_path):
        self.amino_acids = {amino: idx for idx, amino in enumerate(AMINO_ACIDS_TYPES)}
        with open(positive_samples_filepath) as positive_dataset:
            positive_peptides = positive_dataset.read().strip().split('\n')
            positive_peptides = 9 * positive_peptides
            
        with open(negative_samples_file_path) as negative_dataset:
            negative_peptides = negative_dataset.read().strip().split('\n')

        self.peptides = positive_peptides + negative_peptides
        self.labels = torch.tensor([1.0] * len(positive_peptides) + [0.0] * len(negative_peptides), dtype=torch.float)
    
    def peptide_to_indices(self, peptide: str) -> torch.Tensor:
        return torch.tensor(
            [self.amino_acids[ch] for ch in peptide]
        )
    def one_hot_encode(self, peptide: str) -> torch.Tensor:
        return torch.nn.functional.one_hot(
            self.peptide_to_indices(peptide),
            len(self.amino_acids)
        )
    def __len__(self) -> int:
        return len(self.peptides)
    
    def __getitem__(self, index):
        x = self.one_hot_encode(self.peptides[index]).float()
        y = self.labels[index]
        return x, y


