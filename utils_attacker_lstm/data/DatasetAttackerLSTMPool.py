from os import PathLike

import numpy as np
import torch

from utils_io import read_bitarrays
from .DatasetAttackerLSTM import DatasetAttackerLSTM


class DatasetAttackerLSTMPool(DatasetAttackerLSTM):

    def __init__(self,
                 genomes_pool_path: str | bytes | PathLike[str] | PathLike[bytes],
                 genomes_reference_path: str | bytes | PathLike[str] | PathLike[bytes],
                 num_snps: int,
                 dtype: torch.dtype = torch.float32):

        genomes_pool = read_bitarrays(genomes_pool_path)[:, :num_snps]
        genomes_reference = read_bitarrays(genomes_reference_path)[:, :num_snps]
        genomes = np.concatenate((genomes_pool, genomes_reference), axis=0)
        labels_pool = np.ones(genomes_pool.shape[0], dtype=bool)
        labels_reference = np.zeros(genomes_reference.shape[0], dtype=bool)
        labels = np.concatenate((labels_pool, labels_reference), axis=0).astype(bool)
        frequencies_pool = np.mean(genomes_pool, axis=0)
        frequencies_reference = np.mean(genomes_reference, axis=0)

        num_genomes = genomes.shape[0]
        genomes = genomes[..., np.newaxis]
        frequencies_pool = np.broadcast_to(frequencies_pool, (num_genomes, num_snps))[..., np.newaxis]
        frequencies_reference = np.broadcast_to(frequencies_reference, (num_genomes, num_snps))[..., np.newaxis]
        data = np.concatenate((genomes, frequencies_pool, frequencies_reference), axis=2)

        super().__init__(features=data,
                         targets=labels,
                         dtype=dtype)
