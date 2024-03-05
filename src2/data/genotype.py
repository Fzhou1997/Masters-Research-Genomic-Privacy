from pandas import DataFrame


class GenotypeProcessor:
    def __init__(self, raw_path: str, res_path: str, out_path: str, build: int):
        self.raw_path = raw_path
        self.res_path = res_path
        self.out_path = out_path
        self.build = build
        self.genotype_distribution = None
        self.allele_distribution = None

    def set_genotype_distribution(self, genotype_distribution: GenotypeDistribution):
        self.genotype_distribution = genotype_distribution
        pass

    def set_allele_distribution(self, allele_distribution: AlleleDistribution):
        self.allele_distribution = allele_distribution
        pass

    def load_raw(self, user_id: int) -> DataFrame:
        """Loads the raw genotype data for a given user."""
        pass

    def load_processed(self, user_id: int) -> DataFrame:
        """Loads the processed genotype data for a given user."""
        pass

    def save_processed(self, user_id: int, genotype: DataFrame):
        """Saves the processed genotype data for a given user."""
        pass

    def impute(self, genotype: DataFrame) -> DataFrame:
        """Imputes missing genotype and appends the imputed genotype to the dataframe."""
        pass

    def expand(self, genotype: DataFrame) -> DataFrame:
        """Appends individual alleles and allele one hot encoding to the dataframe."""
        pass

    def encode(self, genotype: DataFrame) -> DataFrame:
        """Appends alternate allele counts to the dataframe"""
        pass