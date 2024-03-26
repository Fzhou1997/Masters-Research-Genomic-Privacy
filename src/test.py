from src.genotypes.genotype import Genotype

genotype = Genotype()
genotype.load('../data/genotype/out', 1, 37)
print(genotype.get_one_hot())