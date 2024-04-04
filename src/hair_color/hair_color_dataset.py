import os
import multiprocessing as mp

from enum import IntEnum
from typing import Self

import torch
from torch import Tensor, device
from torch.utils.data import Dataset
from tqdm import tqdm

from genomes import *
from torch_utils import stratified_random_split

AUTOSOMAL = [str(i) for i in range(1, 23)]


# region <phenotypes format>
def _preprocess_phenotypes(in_path, out_path):
    phenotypes = Phenotype()
    phenotypes.from_feature(in_path, 'hair_color')
    phenotypes.clean()
    phenotypes.encode(HAIR_COLOR_ENCODER_READABLE)
    phenotypes.encode(HAIR_COLOR_ENCODER_ORDINAL)
    phenotypes.save(out_path)
    return phenotypes

# endregion </phenotypes format>


# region <genotype format>
def _preprocess_genotype_format(data_path, out_path, res_path, user_id, build, chromosomes):
    try:
        genotype = Genotype()
        genotype.from_user_id(data_path, res_path, user_id, build)
        genotype.clean()
        genotype.filter_rsids_proprietary()
        genotype.filter_chromosomes(chromosomes)
        genotype.save(out_path)
    except FileNotFoundError:
        pass
    except ValueError:
        pass


def _preprocess_genotypes_format_parallel(data_path, out_path, res_path, user_ids, build, chromosomes):
    with mp.Pool(os.cpu_count() - 1) as pool:
        for user_id in user_ids:
            pool.apply_async(_preprocess_genotype_format,
                             args=(data_path, out_path, res_path, user_id, build, chromosomes))
        pool.close()
        pool.join()


# endregion </genotype format>


# region <genotype impute>
def _preprocess_genotype_impute(genotype_out_path, user_id, build, genotypes):
    genotype = Genotype()
    try:
        genotype.load(genotype_out_path, user_id, build)
        genotype.impute_bayesian(genotypes)
        genotype.save(genotype_out_path)
    except FileNotFoundError:
        pass
    except ValueError as e:
        if e == 'No valid rsids found':
            genotype.remove(genotype_out_path)
        pass


def _preprocess_genotypes_impute_parallel(stats_path, genotype_out_path, phenotypes, build):
    phenotype_values = phenotypes.get_values()
    user_ids = list(phenotypes.get_user_ids())
    phenotypes_df = phenotypes.get_phenotypes()
    genotypes_dict = {}
    genotypes_lists = {}
    for phenotype_value in phenotype_values:
        genotypes_dict[phenotype_value] = Genotypes(build)
        genotypes_lists[phenotype_value] = []
    for user_id in tqdm(user_ids):
        try:
            phenotype_value = phenotypes_df.at[user_id, 'hair_color']
            genotype = Genotype()
            genotype.load(genotype_out_path, user_id, build)
            genotypes_lists[phenotype_value].append(genotype)
            if len(genotypes_lists[phenotype_value]) >= 64:
                genotypes_dict[phenotype_value].concat_genotypes(genotypes_lists[phenotype_value])
                genotypes_lists[phenotype_value] = []
        except FileNotFoundError:
            continue
        except ValueError:
            continue
    for phenotype_value in phenotype_values:
        if len(genotypes_lists[phenotype_value]) > 0:
            genotypes_dict[phenotype_value].concat_genotypes(genotypes_lists[phenotype_value])
            genotypes_lists[phenotype_value] = []
    for phenotype_value in phenotype_values:
        genotypes_dict[phenotype_value].save(stats_path)
    with mp.Pool(os.cpu_count() - 1) as pool:
        for user_id in user_ids:
            phenotype_value = phenotypes_df.at[user_id, 'hair_color']
            pool.apply_async(_preprocess_genotype_impute, args=(genotype_out_path, user_id, build, genotypes_dict[phenotype_value]))
        pool.close()
        pool.join()


def _preprocess_genotypes_impute(stats_path, genotype_out_path, phenotypes, build):
    phenotype_values = phenotypes.get_values()
    user_ids = list(phenotypes.get_user_ids())
    phenotypes_df = phenotypes.get_phenotypes()
    genotypes_dict = {}
    genotypes_lists = {}
    for phenotype_value in phenotype_values:
        genotypes_dict[phenotype_value] = Genotypes(build)
        genotypes_lists[phenotype_value] = []
    for user_id in tqdm(user_ids):
        try:
            phenotype_value = phenotypes_df.at[user_id, 'hair_color']
            genotype = Genotype()
            genotype.load(genotype_out_path, user_id, build)
            genotypes_lists[phenotype_value].append(genotype)
            if len(genotypes_lists[phenotype_value]) >= 64:
                genotypes_dict[phenotype_value].concat_genotypes(genotypes_lists[phenotype_value])
                genotypes_lists[phenotype_value] = []
        except FileNotFoundError:
            continue
        except ValueError:
            continue
    for phenotype_value in phenotype_values:
        if len(genotypes_lists[phenotype_value]) > 0:
            genotypes_dict[phenotype_value].concat_genotypes(genotypes_lists[phenotype_value])
            genotypes_lists[phenotype_value] = []
    for phenotype_value in phenotype_values:
        genotypes_dict[phenotype_value].save(stats_path)
    for user_id in user_ids:
        phenotype_value = phenotypes_df.at[user_id, 'hair_color']
        _preprocess_genotype_impute(genotype_out_path, user_id, build, genotypes_dict[phenotype_value])


# endregion </genotype impute>


# region <genotype filter>
def _preprocess_genotype_filter(genotype_out_path, user_id, build, rsids):
    try:
        genotype = Genotype()
        genotype.load(genotype_out_path, user_id, build)
        genotype.filter_rsids(rsids)
        genotype.save(genotype_out_path)
    except FileNotFoundError:
        pass
    except ValueError:
        pass


def _preprocess_genotypes_filter_parallel(stats_path, genotype_out_path, user_ids, build):
    rsids = Rsids(build)
    genotypes_list = []
    for user_id in user_ids:
        try:
            genotype = Genotype()
            genotype.load(genotype_out_path, user_id, build)
            genotypes_list.append(genotype)
            if len(genotypes_list) >= 64:
                rsids.concat_genotypes(genotypes_list)
                genotypes_list = []
        except FileNotFoundError:
            continue
        except ValueError:
            continue
    rsids.save(stats_path)
    common_rsids = rsids.get_common_rsids()
    with mp.Pool(os.cpu_count() - 1) as pool:
        for user_id in user_ids:
            pool.apply_async(_preprocess_genotype_filter, args=(genotype_out_path, user_id, build, common_rsids))
        pool.close()
        pool.join()
    return rsids


def _preprocess_genotypes_filter(stats_path, genotype_out_path, user_ids, build):
    rsids = Rsids(build)
    genotypes_list = []
    for user_id in tqdm(user_ids):
        try:
            genotype = Genotype()
            genotype.load(genotype_out_path, user_id, build)
            genotypes_list.append(genotype)
            if len(genotypes_list) >= 64:
                rsids.concat_genotypes(genotypes_list)
                genotypes_list = []
        except FileNotFoundError:
            continue
        except ValueError:
            continue
    if len(genotypes_list) > 0:
        rsids.concat_genotypes(genotypes_list)
        genotypes_list = []
    rsids.save(stats_path)
    common_rsids = rsids.get_common_rsids()
    for user_id in user_ids:
        _preprocess_genotype_filter(genotype_out_path, user_id, build, common_rsids)
    return rsids
# endregion </genotype filter>


# region <genotype encode>
def _preprocess_genotype_encode(genotype_out_path, user_id, build, reference_alleles):
    try:
        genotype = Genotype()
        genotype.load(genotype_out_path, user_id, build)
        genotype.encode_alternate_allele_count(reference_alleles)
        genotype.save(genotype_out_path)
    except FileNotFoundError:
        pass
    except ValueError:
        pass


def _preprocess_genotypes_encode_parallel(stats_path, genotype_out_path, user_ids, build):
    genotypes = Genotypes(build)
    genotypes_list = []
    for user_id in tqdm(user_ids):
        try:
            genotype = Genotype()
            genotype.load(genotype_out_path, user_id, build)
            genotypes_list.append(genotype)
            if len(genotypes_list) >= 64:
                genotypes.concat_genotypes(genotypes_list)
                genotypes_list = []
        except FileNotFoundError:
            continue
        except ValueError:
            continue
    if len(genotypes_list) > 0:
        genotypes.concat_genotypes(genotypes_list)
        genotypes_list = []
    genotypes.save(stats_path)
    reference_alleles = genotypes.get_reference_alleles()
    with mp.Pool(os.cpu_count() - 1) as pool:
        for user_id in user_ids:
            pool.apply_async(_preprocess_genotype_encode, args=(genotype_out_path, user_id, build, reference_alleles))
        pool.close()
        pool.join()


def _preprocess_genotypes_encode(stats_path, genotype_out_path, user_ids, build):
    genotypes = Genotypes(build)
    genotypes_list = []
    for user_id in tqdm(user_ids):
        try:
            genotype = Genotype()
            genotype.load(genotype_out_path, user_id, build)
            genotypes_list.append(genotype)
            if len(genotypes_list) >= 64:
                genotypes.concat_genotypes(genotypes_list)
                genotypes_list = []
        except FileNotFoundError:
            continue
        except ValueError:
            continue
    if len(genotypes_list) > 0:
        genotypes.concat_genotypes(genotypes_list)
        genotypes_list = []
    genotypes.save(stats_path)
    reference_alleles = genotypes.get_reference_alleles()
    for user_id in user_ids:
        _preprocess_genotype_encode(genotype_out_path, user_id, build, reference_alleles)


# endregion </genotype encode>


# region <genomes format>
def _preprocess_genomes_format(genotype_out_path, genomes_out_path, phenotypes, user_ids, rsids, build):
    genomes = Genomes(build)
    for user_id in tqdm(user_ids):
        try:
            genotype = Genotype()
            genotype.load(genotype_out_path, user_id, build)
            genomes.concat_genotype(genotype)
        except FileNotFoundError:
            continue
        except ValueError:
            continue
    genomes.concat_phenotypes(phenotypes)
    genomes.filter_phenotypes_genotypes()
    common_rsids = list(rsids.get_common_rsids())
    genomes.sort_rsids(common_rsids)
    genomes.save(genomes_out_path)
    return genomes


# endregion </genomes format>


class PreprocessingStage(IntEnum):
    ALL = 0
    PHENOTYPES_FORMAT = 1
    GENOTYPE_FORMAT = 2
    GENOTYPE_IMPUTE = 3
    GENOTYPE_FILTER = 4
    GENOTYPE_ENCODE = 5
    GENOMES_FORMAT = 6
    NONE = 7


class HairColorDataset(Dataset):
    def __init__(self):
        self.data = None
        self.data_length = None
        self.data_classes = None
        self.num_data_classes = None
        self.data_class_counts = None
        self.data_class_weights = None
        self.labels = None
        self.label_length = None
        self.label_classes = None
        self.num_label_classes = None
        self.label_class_counts = None
        self.label_class_weights = None

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def _set_data(self, data: Tensor) -> None:
        self.data = data
        self.data_length = self.data.size(0)
        self.data_classes, self.data_class_counts = torch.unique(self.data, return_counts=True)
        self.num_data_classes = self.data_classes.size(0)
        self.data_class_weights = 1.0 / self.data_class_counts

    def _set_labels(self, labels: Tensor) -> None:
        self.labels = labels
        self.label_length = self.labels.size(0)
        self.label_classes, self.label_class_counts = torch.unique(self.labels, return_counts=True)
        self.num_label_classes = self.label_classes.size(0)
        self.label_class_weights = 1.0 / self.label_class_counts

    def get_data_classes(self) -> Tensor:
        return self.data_classes

    def get_num_data_classes(self) -> int:
        return self.num_data_classes

    def get_data_class_counts(self) -> Tensor:
        return self.data_class_counts

    def get_data_class_weights(self) -> Tensor:
        return self.data_class_weights

    def get_label_classes(self) -> Tensor:
        return self.label_classes

    def get_num_label_classes(self) -> int:
        return self.num_label_classes

    def get_label_class_counts(self) -> Tensor:
        return self.label_class_counts

    def get_label_class_weights(self) -> Tensor:
        return self.label_class_weights

    def split_train_test(self, train_ratio=0.8) -> tuple[Self, Self]:
        train_set = HairColorDataset()
        test_set = HairColorDataset()
        train_data, train_labels, test_data, test_labels = stratified_random_split(self.data, self.labels, train_ratio)
        train_set._set_data(train_data)
        train_set._set_labels(train_labels)
        test_set._set_data(test_data)
        test_set._set_labels(test_labels)
        return train_set, test_set

    def preprocess(self,
                   in_path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
                   out_path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
                   res_path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
                   build: int,
                   chromosomes: list[str]) -> None:
        phenotype_in_path = os.path.join(in_path, '/phenotypes/')
        genotype_in_path = os.path.join(in_path, '/genotypes/')
        phenotype_out_path = os.path.join(out_path, '/phenotypes/')
        genotype_out_path = os.path.join(out_path, '/genotypes/')
        genomes_out_path = os.path.join(out_path, '/genomes/')
        phenotypes = _preprocess_phenotypes(phenotype_in_path, phenotype_out_path)
        _preprocess_genotypes_format_parallel(genotype_in_path, genotype_out_path, res_path, phenotypes, build, chromosomes)
        _preprocess_genotypes_impute_parallel(genomes_out_path, genotype_out_path, phenotypes, build)
        rsids = _preprocess_genotypes_filter(genomes_out_path, genotype_out_path, phenotypes, build)
        _preprocess_genotypes_encode_parallel(genomes_out_path, genotype_out_path, phenotypes, build)
        genomes = _preprocess_genomes_format(genotype_out_path, genomes_out_path, phenotypes, phenotypes, rsids, build)
        self.from_genomes(genomes)

    def from_genomes(self, genomes: Genomes) -> None:
        self._set_data(genomes.get_genotypes_tensor().int())
        self._set_labels(genomes.get_phenotypes_tensor().int())

    def load(self, path: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> None:
        self._set_data(torch.load(os.path.join(path, 'genotypes.pt')))
        self._set_labels(torch.load(os.path.join(path, 'phenotypes.pt')))

    def save(self, path: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.data, os.path.join(path, 'genotypes.pt'))
        torch.save(self.labels, os.path.join(path, 'phenotypes.pt'))

    def to_device(self, device: device) -> None:
        self.data = self.data.to(device)
        self.labels = self.labels.to(device)
