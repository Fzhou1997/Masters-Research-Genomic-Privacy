import os

import numpy as np
import pandas as pd

from src.datasets.genotype_file import GenotypeFile


class Genotype23AndMeLoader:
    def __init__(self, path):
        self.path = path
        self.provider = '23andme'
        self.meta_start_string = 'reference human assembly build'
        self.meta_start_string_len = len(self.meta_start_string)
        self.separator = "\t"
        self.comment = "#"
        self.headers_in = ["rsid", "chromosome", "position", "genotype"]
        self.dtypes_in = {"rsid": str, "chromosome": str, "position": int, "genotype": str}

    def load(self, file_name):
        # region <Error Handling>
        if not file_name.endswith(f'.{self.provider}.txt'):
            raise ValueError(f'File name {file_name} is not provided by {self.provider}.')
        # endregion

        # region <FileIO>
        file_path = os.path.join(self.path, file_name)
        with open(file_path, 'r') as f:
            lines = f.readlines()
        # endregion

        # region <Metadata>
        comments = [line.lstrip('#') for line in lines if line.startswith('#')]
        metadata = ''.join(comments).lower()
        start_string_index = metadata.find(self.meta_start_string)
        version = np.nan
        if start_string_index != -1:
            version_index = start_string_index + self.meta_start_string_len + 1
            try:
                version = int(metadata[version_index:version_index + 2])
            except ValueError:
                version = np.nan
        # endregion

        # region <Data>
        data = pd.read_csv(file_path,
                           sep=self.separator,
                           comment=self.comment,
                           header=None,
                           names=self.headers_in,
                           dtype=self.dtypes_in)
        data['allele1'] = data['genotype'].str[0]
        data['allele2'] = data['genotype'].str[1]
        data = data.drop(columns=['genotype'])
        data.drop_duplicates(inplace=True)
        # endregion

        # region <Data Class Instance>
        genotype = GenotypeFile(file_name, version, data)
        # endregion

        return genotype

