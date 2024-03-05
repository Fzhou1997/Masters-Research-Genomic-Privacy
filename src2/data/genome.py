from pandas import DataFrame


class GenomeProcessor:
    def __init__(self, out_path: str, build: int):
        self.out_path = out_path
        self.build = build
        self.genome = None

    def append(self, user_id: int, genotype: DataFrame, phenotype: str):
        """Appends the genotype data for a given user."""
        pass

    def get(self) -> DataFrame:
        """Returns the current genome data."""
        pass

    def save_processed(self):
        """Saves the current genome data."""
        pass

    def load_processed(self):
        """Loads the current genome data."""
        pass