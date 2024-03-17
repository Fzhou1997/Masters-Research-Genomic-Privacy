import itertools

import pandas as pd

genotype1 = pd.DataFrame(columns=['genotype'])
genotype1.index.name = 'rsid'
genotype1.loc['rs0'] = ['AA']
rsids1 = pd.DataFrame(columns=['chrom', 'pos'])
rsids1.index.name = 'rsid'
rsids1.loc['rs0'] = ['1', 1000]

genotype2 = pd.DataFrame(columns=['genotype'])
genotype2.index.name = 'rsid'
genotype2.loc['rs1'] = ['CC']
rsids2 = pd.DataFrame(columns=['chrom', 'pos'])
rsids2.index.name = 'rsid'
rsids2.loc['rs1'] = ['1', 1001]

genotype3 = pd.DataFrame(columns=['genotype'])
genotype3.index.name = 'rsid'
genotype3.loc['rs0'] = ['GG']
genotype3.loc['rs1'] = ['TT']
rsids3 = pd.DataFrame(columns=['chrom', 'pos'])
rsids3.index.name = 'rsid'
rsids3.loc['rs0'] = ['1', 1000]
rsids3.loc['rs1'] = ['1', 1001]

phenotypes = pd.DataFrame(columns=['phenotype'])
phenotypes.index.name = 'user_id'
phenotypes.loc[1] = [1]
phenotypes.loc[2] = [2]
phenotypes.loc[3] = [3]

genomes = pd.DataFrame()
user1 = genotype1.T
user1.index = pd.Index([1], name='user_id')
genomes = pd.concat([genomes, user1], ignore_index=False)
user2 = genotype2.T
user2.index = pd.Index([2], name='user_id')
genomes = pd.concat([genomes, user2], ignore_index=False)
user3 = genotype3.T
user3.index = pd.Index([3], name='user_id')
genomes = pd.concat([genomes, user3], ignore_index=False)
genomes = pd.concat([genomes, phenotypes], axis=1, ignore_index=False)
print(genomes)

#
# rsids1['count'] = 1
# rsids = pd.DataFrame()
# rsids = pd.concat([rsids, rsids1], ignore_index=False)
# rsids = rsids.groupby(['rsid', 'chrom', 'pos']).sum().reset_index()
# rsids = rsids.set_index('rsid')
# rsids['probability'] = rsids['count'] / 1
# rsids2['count'] = 1
# rsids = pd.concat([rsids, rsids2], ignore_index=False)
# rsids = rsids.groupby(['rsid', 'chrom', 'pos']).sum().reset_index()
# rsids = rsids.set_index('rsid')
# rsids['probability'] = rsids['count'] / 2
# rsids3['count'] = 1
# rsids = pd.concat([rsids, rsids3], ignore_index=False)
# rsids = rsids.groupby(['rsid', 'chrom', 'pos']).sum().reset_index()
# rsids = rsids.set_index('rsid')
# rsids['probability'] = rsids['count'] / 3
# rsids_probability = rsids[['count']] / 3
#
#
# genotypes_df = pd.DataFrame()
# genotypes = [''.join(item) for item in itertools.product('ACGTDI', repeat=2)] + ['--']
# one_hot1 = genotype1.loc[:, []]
# for genotype in genotypes:
#     one_hot1[genotype] = (genotype1['genotype'] == genotype).astype(int)
# genotypes_df = pd.concat([genotypes_df, one_hot1], ignore_index=False)
# genotypes_df = genotypes_df.groupby(level='rsid').sum()
# one_hot2 = genotype2.loc[:, []]
# for genotype in genotypes:
#     one_hot2[genotype] = (genotype2['genotype'] == genotype).astype(int)
# genotypes_df = pd.concat([genotypes_df, one_hot2], ignore_index=False)
# genotypes_df = genotypes_df.groupby(level='rsid').sum()
# one_hot3 = genotype3.loc[:, []]
# for genotype in genotypes:
#     one_hot3[genotype] = (genotype3['genotype'] == genotype).astype(int)
# genotypes_df = pd.concat([genotypes_df, one_hot3], ignore_index=False)
# genotypes_df = genotypes_df.groupby(level='rsid').sum()
# genotype_probabilties = genotypes_df.div(genotypes_df.sum(axis=1), axis=0)
#
# ALLELES = 'ACGTDI-'
# GENOTYPES = [''.join(item) for item in itertools.product('ACGTDI', repeat=2)] + ['--']
# TRANSFORMATION = pd.DataFrame(columns=list(ALLELES), index=GENOTYPES)
# for genotype in GENOTYPES:
#     for allele in ALLELES:
#         TRANSFORMATION.at[genotype, allele] = genotype.count(allele)
#
# allele_counts = genotypes_df.dot(TRANSFORMATION)
# reference_alleles = allele_counts.idxmax(axis=1).to_frame(name='allele')
# print(reference_alleles)