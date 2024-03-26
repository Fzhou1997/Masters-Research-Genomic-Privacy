import os

import multiprocessing as mp

from src3.genotypes.genotype import Genotype
from src3.genotypes.phenotypes import Phenotypes, convert_hair_colors
from src3.genotypes.rsids import Rsids


def _genotype_from_user_id(genomes_queue, genotype_data_path, genotype_out_path, res_path, user_id, hair_color):
    genotype = Genotype()
    try:
        genotype.from_user_id(genotype_data_path, res_path, user_id, 37)
    except Exception as e:
        print(f'Error loading genotype for user {user_id}: {e}')
        return
    try:
        genotype.filter_chromosomes(set(str(i) for i in range(1, 23)))
    except Exception as e:
        print(f'Error filtering chromosomes for user {user_id}: {e}')
        return
    genotype.save(genotype_out_path)
    genomes_queue.put((user_id, {'genotype': genotype, 'phenotype': hair_color}))


def _genotype_load(genomes_queue, genotype_out_path, user_id, hair_color):
    genotype = Genotype()
    try:
        genotype.load(genotype_out_path, user_id, 37)
    except FileNotFoundError:
        return
    except Exception as e:
        print(f'Error loading genotype for user {user_id}: {e}')
        return
    genomes_queue.put((user_id, {'genotype': genotype, 'phenotype': hair_color}))


def _genotype_filter_rsids(genotype, genotype_out_path, rsids):
    genotype.filter_rsids(rsids)
    genotype.save(genotype_out_path)


class GenomesHairColor:
    def __init__(self):
        self.genomes = None  # dataframe of user_ids to rsids and hair color
        self.genotype_counts = {'blonde': None, 'brown': None, 'black': None}  # set of feature_values to Genotypes objects
        self.rsid_counts = None  # Rsids object

    def from_opensnps(self, genotype_data_path, phenotype_data_path, genotype_out_path, phenotype_out_path, counts_out_path, res_path):
        phenotypes = Phenotypes()
        phenotypes.from_feature(phenotype_data_path, 'Hair Color', converter=convert_hair_colors)
        phenotypes.save(phenotype_out_path)
        phenotypes_df = phenotypes.get_phenotypes()
        user_ids = phenotypes.get_user_ids()
        manager = mp.Manager()
        genomes_queue = manager.Queue()
        with mp.Pool(os.cpu_count() - 1) as pool:
            results = []
            for user_id in user_ids:
                result = pool.apply_async(_genotype_from_user_id, args=(genomes_queue, genotype_data_path, genotype_out_path, res_path, user_id, phenotypes_df.loc[user_id, 'hair_color']))
                results.append(result)
            pool.close()
            pool.join()
        for result in results:
            try:
                result.get()
            except Exception as e:
                print(f'Error in child process: {e}')
        while not genomes_queue.empty():
            (user_id, genome) = genomes_queue.get()
            self.genomes[user_id] = genome
        self.rsid_counts = Rsids()
        for user_id in self.genomes:
            self.rsid_counts.append(self.genomes[user_id]['genotype'].get_rsids())
        common_rsids = self.rsid_counts.get_common_rsids()
        with mp.Pool(os.cpu_count() - 1) as pool:
            results = []
            for user_id in self.genomes:
                result = pool.apply_async(_genotype_filter_rsids, args=(self.genomes[user_id]['genotype'], genotype_out_path, common_rsids))
                results.append(result)
            pool.close()
            pool.join()

    def resume(self, genotype_out_path, phenotype_out_path, counts_out_path, from_stage=0):
        phenotypes = Phenotypes()
        phenotypes.load(phenotype_out_path, "hair_color")
        phenotypes_df = phenotypes.get_phenotypes()
        user_ids = phenotypes.get_user_ids()
        manager = mp.Manager()
        genomes_queue = manager.Queue()
        with mp.Pool(os.cpu_count() - 1) as pool:
            results = []
            for user_id in user_ids:
                result = pool.apply_async(_genotype_load, args=(genomes_queue, genotype_out_path, user_id, phenotypes_df.loc[user_id, 'hair_color']))
                results.append(result)
            pool.close()
            pool.join()
        for result in results:
            try:
                result.get()
            except Exception as e:
                print(f'Error in child process: {e}')
        while not genomes_queue.empty():
            (user_id, genome) = genomes_queue.get()
            self.genomes[user_id] = genome
        self.rsid_counts = Rsids()
        if from_stage <= 0:
            for user_id in self.genomes:
                self.rsid_counts.append(self.genomes[user_id]['genotype'].get_rsids())
            self.rsid_counts.save(counts_out_path)
        else:
            self.rsid_counts.load(counts_out_path)
        if from_stage <= 1:
            optimal_rsids = self.rsid_counts.get_common_rsids()
            with mp.Pool(os.cpu_count() - 1) as pool:
                results = []
                for user_id in self.genomes:
                    result = pool.apply_async(_genotype_filter_rsids, args=(self.genomes[user_id]['genotype'], genotype_out_path, optimal_rsids))
                    results.append(result)
                pool.close()
                pool.join()
            for result in results:
                try:
                    result.get()
                except Exception as e:
                    print(f'Error in child process: {e}')



if __name__ == '__main__':
    genomes = GenomesHairColor()
    genomes.from_opensnps('../data/genotype/raw', '../data/phenotype/raw', '../data/genotype/out', '../data/phenotype/out', '../data/stats', '../res')
    # genomes.resume('../data/genotype/out', '../data/phenotype/out', '../data/stats', 0)

