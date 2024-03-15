import argparse
import os

from legacy.src2.data import loaders


def make_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build",
        help="Assembly build to use, usually can be 36/37/38",
        default=37, type=lambda value: int(value))
    parser.add_argument(
        "--raw",
        help="Directory where raw genotype files are stored",
        default='data/opensnp/raw')
    parser.add_argument(
        "--out",
        help="Directory where output allele files are stored",
        default='data/opensnp/out')
    parser.add_argument(
        "--res",
        help="Directory where resource files are stored",
        default='data/res')
    return parser


if __name__ == '__main__':
    parser = make_arg_parser()
    args = parser.parse_args()

    # get the unique user files from our phenotype loader
    # - this step ensures we only look at files with phenotype labels
    phenotype_loader = loaders.PhenotypeLoader(
        args.raw, id=202402130100, verbose=True)
    filenames = [
        phenotype_loader.get_filename(user_id)
        for user_id in phenotype_loader.df.index
    ]
    hair_colors = phenotype_loader.df.hair_color

    # load in common RSIDs to trim genotype files with
    with open(os.path.join(args.out, f'common_rsids_build{args.build}.txt'), 'r') as f:
        rsids = {rsid.strip() for rsid in f.readlines()}

    snps_loader = loaders.SNPSLoader(args.raw, res_path=args.res)

    # load in our allele distribution
    distribution = legacy.src2.dist.allele.from_csv(args.out, args.build)
    if not distribution:
        distributions = legacy.src2.dist.allele.from_filenames(
            filenames, phenotype_loader.df.hair_color, snps_loader, verbose=True, rsids=rsids)
        for build in distributions:
            distributions[build].save(args.out)
        distribution = distributions[args.build]

    """
    rsid,chrom,pos,hair_color,A,C,G,T,D,I,-
    rs1000007,2,237416793,black,0,5,0,17,0,0,0
    rs1000007,2,237416793,blonde,0,1,0,7,0,0,0
    rs1000007,2,237416793,brown,0,37,0,99,0,0,0
    rs1000007,2,237752054,brown,0,0,0,2,0,0,0
    rs10000226,4,87738967,brown,0,1,0,1,0,0,0
    rs10000226,4,87957991,black,0,21,2,1,0,0,0
    rs10000226,4,87957991,blonde,0,7,0,1,0,0,0
    rs10000226,4,87957991,brown,0,123,0,13,0,0,0
    rs1000031,18,44615439,black,10,2,12,0,0,0,0
    """

    # >>> here



    # find the alleles that occur the most for each RSID
    reference_alleles = distribution.data\
        .groupby('rsid')[['A', 'C', 'G', 'T', 'D', 'I']]\
        .sum()\
        .idxmax(axis=1)

    # AA, AG, ... -> 0 alt allele, 1, or 2...
    # P(AG or AC or AT) = P(AG) + P(AC) + P(AT) = P(A occuring 1 time) = P(1)
    # compute user x alternate allele count matrix

    # # collect formatted user vectors
    # matrix_filename = os.path.join(args.out, f'matrix_build{args.build}.csv')
    # try:
    #     matrix = pd.read_csv(matrix_filename, index_col=0)
    # except:
    #     user_ids = []
    #     user_vectors = []
    #     for user_id in tqdm(phenotype_loader.df.index):
    #         filename = phenotype_loader.get_filename(user_id)
    #         res = snps_loader.load(filename, expand_alleles=True)
    #         if res is None:
    #             continue
    #         snps, build = res
    #         if build != args.build:
    #             continue
    #         # trim down by common RSIDs!
    #         snps = snps.loc[list(set(snps.index) & stat)]
    #
    #         user_vector = []
    #         for rsid, allele in zip(reference_alleles.index, reference_alleles.values):
    #             # -1 - missing data
    #             #  0 - all reference
    #             #  1 - one alternate
    #             #  2 - two alternates
    #             if rsid not in snps.index or snps.loc[rsid, '-'] > 0:
    #                 user_vector.append(-1)
    #             else:
    #                 user_vector.append(2 - snps.loc[rsid, allele])
    #         user_vector = pd.Series(
    #             user_vector, index=reference_alleles.index, name=user_id)
    #         user_ids.append(user_id)
    #         user_vectors.append(user_vector)
    #
    #     matrix = pd.concat(user_vectors, axis=1).T
    #     matrix.to_csv(os.path.join(args.out, f'matrix_build{args.build}.csv'))
    #
    # hair_colors = phenotype_loader.df.loc[matrix.index].hair_color
    #
    # # make our probability matrices
    # context = dict()
    # for hair_color in hair_colors.unique():
    #     context[hair_color] = (matrix[hair_colors == hair_color] != -1).sum()
    # probabilities = dict()
    # for hair_color in hair_colors.unique():
    #     probabilities[hair_color] = dict()
    #     for n_alternate in range(0, 3):
    #         probabilities[hair_color][n_alternate] =\
    #             (matrix[hair_colors == hair_color] == n_alternate).sum() \
    #             / context[hair_color]

## p(0, black) for each RSID
# 0, 1, 2,
#probs['black'] -> 0.33, 0.1, 0.5

"""
Allele-wise imputation:
Model:
3 (hair color) x 45k (stat) x 36 (possible allele pairs, e.g. "AA", "AC", "AG", etc) matrix of probabilities
Predict:
model[hair_color][rsid]: a list of probabilities for allele pairs "AA", "AC", "AG", etc
random.choice(c=allele_pairs, p=model[hair_color][rsid])

AC = CA? probably not, so it's a 6 * 6 = 36 pairs
A- or -C or other bad pair: we should convert these to "--"


p(allele|hair) = p(0|black) or p(1|black) or ...
               = p(rsid[000000]allele[:]=="aa" | hair=="black") or ...
 = #(allele,hair) / #(hair)

0.9, 0.05, 0.05
0.33, 0.33, 0.33

"""

