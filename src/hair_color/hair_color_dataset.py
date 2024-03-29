import os
import multiprocessing as mp

from enum import IntEnum

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from genomes import *

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
    rsids.sort_rsids()
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
        self.genotypes_tensor = None
        self.phenotypes_tensor = None
        self.class_weights = None

    def __len__(self):
        return self.genotypes_tensor.shape[0]

    def __getitem__(self, idx):
        return self.genotypes_tensor[idx], self.phenotypes_tensor[idx]

    def get_class_weights(self):
        return self.class_weights

    def preprocess(self, in_path, out_path, res_path, build, chromosomes, stage=PreprocessingStage.ALL):
        phenotype_in_path = os.path.join(in_path, '/phenotypes/')
        genotype_in_path = os.path.join(in_path, '/genotypes/')
        phenotype_out_path = os.path.join(out_path, '/phenotypes/')
        genotype_out_path = os.path.join(out_path, '/genotypes/')
        genomes_out_path = os.path.join(out_path, '/genomes/')
        if stage <= PreprocessingStage.PHENOTYPES_FORMAT:
            phenotypes = _preprocess_phenotypes(phenotype_in_path, phenotype_out_path)
        else:
            phenotypes = Phenotype()
            phenotypes.load(phenotype_out_path)
        if stage <= PreprocessingStage.GENOTYPE_FORMAT:
            _preprocess_genotypes_format_parallel(genotype_in_path, genotype_out_path, res_path, phenotypes.get_user_ids(), self.build,
                                                  self.chromosomes)
        if stage <= PreprocessingStage.GENOTYPE_IMPUTE:
            _preprocess_genotypes_impute_parallel(stats_path, genotype_out_path, phenotypes, self.build)
        if stage <= PreprocessingStage.GENOTYPE_FILTER:
            rsids = _preprocess_genotypes_filter(stats_path, genotype_out_path, user_ids, self.build)
        else:
            rsids = Rsids(self.build)
            rsids.load(stats_path)
        if stage <= PreprocessingStage.GENOTYPE_ENCODE:
            _preprocess_genotypes_encode_parallel(stats_path, genotype_out_path, user_ids, self.build)
        self.genomes = _preprocess_genomes_format(genotype_out_path, genomes_out_path, phenotypes, user_ids, rsids, self.build)
        self.genotypes_tensor = self.genomes.get_genotypes_tensor().float()
        self.phenotypes_tensor = self.genomes.get_phenotypes_tensor().float()
        assert self.genotypes_tensor.shape[0] == self.phenotypes_tensor.shape[0]

    def load(self, in_path):
        self.genotypes_tensor = torch.load(os.path.join(in_path, 'genotypes.pt'))
        self.phenotypes_tensor = torch.load(os.path.join(in_path, 'phenotypes.pt'))
        classes, counts = torch.unique(self.phenotypes_tensor, return_counts=True)
        self.class_weights = torch.tensor(1.0 / counts, dtype=torch.float)

    def save(self, out_path):
        torch.save(self.genotypes_tensor, os.path.join(out_path, 'genotypes.pt'))
        torch.save(self.phenotypes_tensor, os.path.join(out_path, 'phenotypes.pt'))

    def to_device(self, device):
        self.genotypes_tensor = self.genotypes_tensor.to(device)
        self.phenotypes_tensor = self.phenotypes_tensor.to(device)
        self.class_weights = self.class_weights.to(device)
