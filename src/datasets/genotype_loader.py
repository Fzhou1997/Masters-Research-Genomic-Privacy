import os

from src.datasets.genotype_23_and_me import Genotype23AndMeLoader
from src.datasets.genotype_ancestry import GenotypeAncestryLoader
from src.datasets.genotype_ftdna_illumina import GenotypeFtdnaIlluminaLoader


class GenotypeLoader:
    def __init__(self, path):
        self.path = path
        self.providers = {
            '23andme': Genotype23AndMeLoader(self.path),
            'ancestry': GenotypeAncestryLoader(self.path),
            'ftdna-illumina': GenotypeFtdnaIlluminaLoader(self.path)
        }

    def load(self, user_id, version):
        # region <FileIO>
        file_names = os.listdir(self.path)
        file_names = [fn for fn in file_names if fn.startswith(f'user{user_id}_')]
        # endregion

        # region <Data>
        genotype_files = []
        for file_name in file_names:
            provider = file_name.split('.')[-2]
            if provider not in self.providers.keys():
                continue
            genotype_files.append(self.providers[provider].load(file_name))
        # endregion

        # region <Filter>
        genotype_files = [gf for gf in genotype_files if gf.version == version]
        genotype_file = max(genotype_files, key=lambda gf: gf.file_id)
        # endregion

        return genotype_file
