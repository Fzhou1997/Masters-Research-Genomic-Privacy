from torch.utils.data import DataLoader


class HairColorDataLoader(DataLoader):
    def __iter__(self):
        for genotype, phenotype in super(HairColorDataLoader, self).__iter__():
            genotype = genotype.unsqueeze(2)
            yield genotype, phenotype

