from os import PathLike

import numpy as np
import torch

from src.utils_io import read_bitarrays
from .DatasetAttackerLSTM import DatasetAttackerLSTM


class DatasetAttackerLSTMBeacon(DatasetAttackerLSTM):

    def __init__(self,
                 genomes_beacon_path: str | bytes | PathLike[str] | PathLike[bytes],
                 genomes_reference_path: str | bytes | PathLike[str] | PathLike[bytes],
                 num_snps: int,
                 dtype: torch.dtype = torch.float32):
        genomes_beacon = read_bitarrays(genomes_beacon_path)[:, :num_snps]
        genomes_reference = read_bitarrays(genomes_reference_path)[:, :num_snps]
        genomes = np.concatenate((genomes_beacon, genomes_reference), axis=0)
        labels_beacon = np.ones(genomes_beacon.shape[0], dtype=bool)
        labels_reference = np.zeros(genomes_reference.shape[0], dtype=bool)
        labels = np.concatenate((labels_beacon, labels_reference), axis=0).astype(bool)
        presences_beacon = np.any(genomes_beacon, axis=0).astype(bool)
        frequencies_reference = np.mean(genomes, axis=0)

        num_genomes = genomes.shape[0]
        genomes = genomes[..., np.newaxis]
        presences_beacon = np.broadcast_to(presences_beacon, (num_genomes, num_snps))[..., np.newaxis]
        frequencies_reference = np.broadcast_to(frequencies_reference, (num_genomes, num_snps))[..., np.newaxis]
        data = np.concatenate((genomes, presences_beacon, frequencies_reference), axis=2)

        super().__init__(features=data,
                         targets=labels,
                         dtype=dtype)
