from collections import Counter

from genomes import Genotype


class Summary:
    def __init__(self, build: int):
        self.build = build
        self.num_genotypes = 0
        self.rsid_map = {}
        self.rsid_counts = Counter()
        self.genotype_counts = {}

    def concat_genotype(self, genotype: Genotype) -> None:
        if genotype.get_build() != self.build:
            raise ValueError('Reference genome build mismatch')
        self.num_genotypes += 1
        rsids = genotype.get_rsids()
        self.rsid_counts.update(rsids.index)
        self.rsid_map.update(rsids.to_dict(orient='index'))
        for rsid, genotype in genotype.get_genotype().iterrows():
            if rsid not in self.genotype_counts:
                self.genotype_counts[rsid] = Counter()
            self.genotype_counts[rsid].update(genotype)

    def concat_genotypes(self, genotypes: list[Genotype]) -> None:
        for genotype in genotypes:
            self.concat_genotype(genotype)

