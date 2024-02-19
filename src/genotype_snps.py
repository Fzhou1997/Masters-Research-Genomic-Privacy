import os

from snps import SNPs

import genomes

def load_raw(data_path, user_ids, build, chroms):
    genome_list = genomes.load_raw(data_path)
    genome_list = genome_list[genome_list['user_id'].isin(user_ids)]
    genotypes = {}
    for genome in genome_list:
        file_name = f"user{genome['user_id']}_file{genome['file_id']}_yearofbirth_{genome['year_of_birth']}_sex_{genome['sex']}.{genome['provider']}.txt"
        file_path = os.path.join(data_path, file_name)
        snps = SNPs(file_path)
        if snps.build != build:
            snps.remap(build)
        if genome['user_id'] not in genotypes:
            genotypes[genome['user_id']] = snps
        else:
            genotypes[genome['user_id']].merge(snps_objects=[snps], remap=True)
        genotypes[genome['user_id']].sort()
        
