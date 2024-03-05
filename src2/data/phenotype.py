from pandas import DataFrame


class PhenotypeProcessor:
    def __init__(self, raw_path: str, out_path: str):
        self.raw_path = raw_path
        self.out_path = out_path
        self.data = None

    def load_raw(self) -> DataFrame:
        """Loads the raw phenotype data."""
        pass

    def load_processed(self) -> DataFrame:
        """Loads the processed phenotype data."""
        pass

    def save_processed(self, phenotype: DataFrame):
        """Saves the processed phenotype data."""
        pass

    def filter(self, user_ids: list[int]) -> DataFrame:
        """Filters the phenotype data for a given list of user IDs."""
        pass