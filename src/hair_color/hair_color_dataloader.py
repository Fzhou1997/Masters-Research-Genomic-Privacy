from torch.utils.data import DataLoader


class HairColorDataLoader(DataLoader):
    def __iter__(self):
        for genotype, phenotype in super(HairColorDataLoader, self).__iter__():
            genotype = genotype.unsqueeze(2)
            yield genotype, phenotype

# genotype, phenotype
# genotype: [0, 2, 1, 0, 0 ,1... etc]
# genotype_time: [[0], [2], [1], [0]...
# phenotype: [0, 1, 0]