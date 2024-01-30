import os

import numpy as np
import pandas as pd

from src.datasets.genotype_file import GenotypeFile


class GenotypeAncestryLoader:
    def __init__(self, path):
        self.path = path
        self.provider = 'ancestry'
        self.meta_start_string = 'human reference build'
        self.meta_start_string_len = len(self.meta_start_string)
        self.separator = "\t"
        self.comment = "#"
        self.headers_in = ["rsid", "chromosome", "position", "allele1", "allele2"]
        self.dtypes_in = {"rsid": str, "chromosome": str, "position": int, "allele1": str,
                          "allele2": str}

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
                           dtype=self.dtypes_in)
        data['allele1'] = data['allele1'].replace('0', '-')
        data['allele2'] = data['allele2'].replace('0', '-')
        data['chromosome'] = data['chromosome'].replace('23', 'X').replace('24', 'Y').replace('25',
                                                                                              'PAR').replace(
            '26', 'MT')
        data.drop_duplicates(inplace=True)
        # endregion

        # region <Data Class Instance>
        genotype = GenotypeFile(file_name, version, data)
        # endregion

        return genotype
