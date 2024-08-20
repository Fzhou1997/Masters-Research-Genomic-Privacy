import os
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from utils_genomes import *
from src.hair_color.dataset import HairColorDataset

num_threads = os.cpu_count() - 1

res_path = 'res/'
phenotypes_in_path = '../data/opensnps/phenotypes/'
genotype_in_path = '../data/opensnps/genotypes/'

phenotypes_out_path = '../data/hair_color/build37_autosomal/phenotypes/'
rsids_out_path = '../data/hair_color/build37_autosomal/rsids/'
genotypes_out_path = '../data/hair_color/build37_autosomal/genotypes/'
genomes_out_path = '../data/hair_color/build37_autosomal/genomes/'
genotype_formatted_out_path = '../data/hair_color/build37_autosomal/genotype/formatted'
genotype_filtered_out_path = '../data/hair_color/build37_autosomal/genotype/filtered'
genotype_imputed_out_path = '../data/hair_color/build37_autosomal/genotype/imputed'
genotype_encoded_out_path = '../data/hair_color/build37_autosomal/genotype/encoded'
dataset_out_path = '../data/hair_color/build37_autosomal/dataset/'

phenotypes_out_file_name = 'phenotypes'
rsids_out_file_name = 'rsids'
genotypes_all_out_file_name = 'genotypes_all'
genotypes_blonde_out_file_name = 'genotypes_blonde'
genotypes_brown_out_file_name = 'genotypes_brown'
genotypes_black_out_file_name = 'genotypes_black'
genomes_out_file_name = 'genomes'
genotype_out_file_name = 'genotype'
dataset_out_file_name = 'dataset'

build = 37
chromosomes = [str(i) for i in range(1, 23)]


def format_genotype(user_id: int):
    try:
        genotype = Genotype()
        genotype.from_user_id(genotype_in_path, res_path, user_id, build)
        genotype.clean()
        genotype.filter_rsids_proprietary()
        genotype.filter_chromosomes(chromosomes)
        genotype.save(genotype_formatted_out_path, f"{genotype_out_file_name}{user_id}")
    except FileNotFoundError or ValueError:
        pass


def filter_genotype(user_id: int, common_rsids: set[str]):
    try:
        genotype = Genotype()
        genotype.load(genotype_formatted_out_path, f"{genotype_out_file_name}{user_id}")
        genotype.filter_rsids(common_rsids)
        genotype.drop_rsid_map()
        genotype.save(genotype_filtered_out_path, f"{genotype_out_file_name}{user_id}")
    except FileNotFoundError or ValueError:
        pass


def impute_genotype(user_id: int, mode_genotypes: dict[str, str | int]):
    try:
        genotype = Genotype()
        genotype.load(genotype_filtered_out_path, f"{genotype_out_file_name}{user_id}")
        genotype.impute_bayesian(mode_genotypes)
        genotype.save(genotype_imputed_out_path, f"{genotype_out_file_name}{user_id}")
    except FileNotFoundError or ValueError:
        pass


def encode_genotype(user_id: int, reference_alleles: dict[str, str]):
    try:
        genotype = Genotype()
        genotype.load(genotype_imputed_out_path, f"{genotype_out_file_name}{user_id}")
        genotype.encode_alternate_allele_count(reference_alleles)
        genotype.save(genotype_encoded_out_path, f"{genotype_out_file_name}{user_id}")
    except FileNotFoundError or ValueError:
        pass


if __name__ == '__main__':
    phenotypes = Phenotype()
    phenotypes.from_feature(phenotypes_in_path, 'hair_color')
    phenotypes.clean()
    phenotypes.encode(HAIR_COLOR_ENCODER_READABLE)
    phenotypes.encode(HAIR_COLOR_ENCODER_ORDINAL)
    phenotypes.save(phenotypes_out_path, phenotypes_out_file_name)

    user_ids = phenotypes.get_user_ids()

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        pool.map(format_genotype, user_ids)

    rsids = Rsids(build)
    for user_id in tqdm(user_ids):
        try:
            genotype = Genotype()
            genotype.load(genotype_formatted_out_path, f"{genotype_out_file_name}{user_id}")
            rsids.concat_genotype(genotype)
        except FileNotFoundError or ValueError:
            continue

    rsids.save(rsids_out_path, rsids_out_file_name)
    common_rsids = rsids.get_common_rsids()
    sorted_rsids = rsids.get_sorted_rsids()
    sorted_rsids = [rsid for rsid in sorted_rsids if rsid in common_rsids]
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        pool.map(lambda user_id: filter_genotype(user_id, common_rsids), user_ids)

    genotypes = {
        0: Genotypes(build),
        1: Genotypes(build),
        2: Genotypes(build)
    }
    for user_id in tqdm(user_ids):
        try:
            genotype = Genotype()
            genotype.load(genotype_filtered_out_path, f"{genotype_out_file_name}{user_id}")
            phenotype = phenotypes[user_id]
            genotypes[phenotype].concat_genotype(genotype)
        except FileNotFoundError or ValueError:
            continue
    genotypes[0].save(genotypes_out_path, genotypes_blonde_out_file_name)
    genotypes[1].save(genotypes_out_path, genotypes_brown_out_file_name)
    genotypes[2].save(genotypes_out_path, genotypes_black_out_file_name)
    mode_genotypes = {
        0: genotypes[0].get_mode_genotypes(),
        1: genotypes[1].get_mode_genotypes(),
        2: genotypes[2].get_mode_genotypes()
    }
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        pool.map(lambda user_id: impute_genotype(user_id, mode_genotypes[phenotypes[user_id]]), user_ids)

    genotypes_all = Genotypes(build)
    for user_id in tqdm(user_ids):
        try:
            genotype = Genotype()
            genotype.load(genotype_imputed_out_path, f"{genotype_out_file_name}{user_id}")
            genotypes_all.concat_genotype(genotype)
        except FileNotFoundError or ValueError:
            continue
    genotypes_all.save(genotypes_out_path, genotypes_all_out_file_name)
    reference_alleles = genotypes_all.get_reference_alleles()
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        pool.map(lambda user_id: encode_genotype(user_id, reference_alleles), user_ids)

    genomes = Genomes(build)
    for user_id in tqdm(user_ids):
        try:
            genotype = Genotype()
            genotype.load(genotype_encoded_out_path, f"{genotype_out_file_name}{user_id}")
            genomes.concat_genotype(genotype)
        except FileNotFoundError or ValueError:
            continue
    genomes.concat_phenotypes(phenotypes)
    genomes.filter_phenotypes_genotypes()
    genomes.sort_rsids(sorted_rsids)
    genomes.save(genomes_out_path, genomes_out_file_name)

    dataset = HairColorDataset()
    dataset.from_genomes(genomes)
    dataset.save(dataset_out_path, dataset_out_file_name)
