import itertools
import json
import os
import warnings
from typing import Self

import pandas as pd
from snps import SNPs

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

RSIDS_INVALID = r'[^0-9a-z]+'
CHROMOSOMES_AUTOSOMAL = set(str(i) for i in range(1, 23))
CHROMOSOMES_INVALID = r'[^0-9XYMT]+'
ALLELES = 'ACGTDI-'
ALLELES_INVALID = r'[^ACGTDI\-0]+'
ALLELES_NA = r'.*[-0]+.*'
GENOTYPES = [''.join(item) for item in itertools.product('ACGT', repeat=2)] + [''.join(item) for item in itertools.product('DI', repeat=2)] + ['--']


class Genotype:
    def __init__(self):
        self.user_id = None
        self.build = None
        self.genotype = None

    def __iter__(self):
        return GenotypeIterator(self)

    def __getitem__(self, idx: str) -> str | None:
        if idx not in self.genotype.index:
            return None
        else:
            return self.genotype.at[idx, 'genotype']

    def _to_dict(self) -> dict[str, int | dict[str, str | int]]:
        return {
            'user_id': self.user_id,
            'build': self.build,
            'genotype': self.genotype.to_dict(orient='index'),
        }

    def _from_dict(self, data: dict[str, int | dict[str, str | int]]) -> Self:
        self.user_id = data['user_id']
        self.build = data['build']
        self.genotype = pd.DataFrame.from_dict(data['genotype'], orient='index')
        self.genotype.index.name = 'rsid'
        return self

    def from_user_id(self,
                     data_path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
                     res_path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
                     user_id: int,
                     build: int) -> Self:
        user_file_names = [file_name for file_name in os.listdir(data_path)
                           if file_name.startswith(f'user{user_id}_')
                           and file_name.endswith('.txt')
                           and 'exome-vcf' not in file_name]
        if not user_file_names:
            raise FileNotFoundError(f'No genotype files found')
        snps = None
        for file_name in user_file_names:
            file_path = os.path.join(data_path, file_name)
            try:
                s = SNPs(file_path, resources_dir=res_path)
            except Exception:
                continue
            if not s.valid or not s.build_detected or s.build != build:
                continue
            if snps is not None:
                s.merge(snps_objects=[snps])
            snps = s
        if snps is None:
            raise ValueError('No valid genotype files found')
        snps.sort()
        self.user_id = user_id
        self.build = build
        self.genotype = snps.snps
        return self

    def clean(self) -> Self:
        self.genotype['genotype'] = self.genotype['genotype'].str.upper()
        self.genotype['genotype'] = self.genotype['genotype'].str.replace(ALLELES_INVALID, '', regex=True)
        self.genotype['genotype'] = self.genotype['genotype'].str.replace(ALLELES_NA, '--', regex=True)
        self.genotype['genotype'] = self.genotype['genotype'].fillna('--')
        self.genotype['chrom'] = self.genotype['chrom'].str.upper()
        self.genotype['chrom'] = self.genotype['chrom'].str.replace(CHROMOSOMES_INVALID, '', regex=True)
        self.genotype.index = self.genotype.index.str.lower()
        self.genotype.index = self.genotype.index.str.replace(RSIDS_INVALID, '', regex=True)
        unique_rsids = ~self.genotype.index.duplicated(keep='first')
        self.genotype = self.genotype[unique_rsids]
        if len(self.genotype.index) == 0:
            raise ValueError('No valid rsids found')
        return self

    def drop_rsid_map(self) -> Self:
        self.genotype = self.genotype.drop(columns=['chrom', 'pos'])
        return self

    def filter_chromosomes_autosomal(self):
        self.genotype = self.genotype[self.genotype['chrom'].isin(CHROMOSOMES_AUTOSOMAL)]
        if len(self.genotype.index) == 0:
            raise ValueError('No valid rsids found')

    def filter_chromosomes(self, chromosomes: list[str]) -> None:
        self.genotype = self.genotype[self.genotype['chrom'].isin(chromosomes)]
        if len(self.genotype.index) == 0:
            raise ValueError('No valid rsids found')

    def filter_rsids_proprietary(self):
        self.genotype = self.genotype.loc[self.genotype.index.str.startswith('rs')]
        if len(self.genotype.index) == 0:
            raise ValueError('No valid rsids found')

    def filter_rsids(self, rsids: set[str]) -> None:
        self.genotype = self.genotype.loc[self.genotype.index.isin(rsids)]
        if len(self.genotype.index) == 0:
            raise ValueError('No valid rsids found')

    def is_valid(self) -> bool:
        return len(self.genotype.index) > 0

    def is_imputed(self) -> bool:
        return '--' not in self.genotype['genotype'].values

    def is_alternate_allele_count_encoded(self):
        return self.genotype['genotype'].isin(range(3)).all()

    def get_user_id(self) -> int:
        return self.user_id

    def get_build(self) -> int:
        return self.build

    def get_rsids(self) -> pd.DataFrame:
        return self.genotype[['chrom', 'pos']]

    def get_genotype(self) -> pd.DataFrame:
        return self.genotype[['genotype']]

    def get_one_hot(self) -> pd.DataFrame:
        one_hot = self.genotype.loc[:, []]
        for genotype in GENOTYPES:
            one_hot[genotype] = (self.genotype['genotype'] == genotype).astype(int)
        return one_hot

    def get_allele_counts(self) -> pd.DataFrame:
        allele_counts = self.genotype.loc[:, []]
        for allele in ALLELES:
            allele_counts[allele] = self.genotype['genotype'].apply(lambda genotype: genotype.count(allele))
        return allele_counts

    def impute_bayesian(self, mode_genotypes: dict[str, str | int]) -> None:
        target_rsids = self.genotype[self.genotype['genotype'] == '--'].index
        for rsid in target_rsids:
            self.genotype.at[rsid, 'genotype'] = mode_genotypes[rsid]

    def encode_alternate_allele_count(self, reference_alleles: dict[str, str]) -> None:
        for rsid in self.genotype.index:
            reference_allele = reference_alleles[rsid]
            if self.genotype.at[rsid, 'genotype'] == '--':
                self.genotype.at[rsid, 'genotype'] = -1
            else:
                self.genotype.at[rsid, 'genotype'] = 2 - self.genotype.at[rsid, 'genotype'].count(reference_allele)

    def save(self, out_path: str | bytes | os.PathLike[str] | os.PathLike[bytes], file_name: str) -> None:
        os.makedirs(out_path, exist_ok=True)
        with open(f'{out_path}/{file_name}.json', 'w') as file:
            json.dump(self._to_dict(), file)

    def load(self, in_path: str | bytes | os.PathLike[str] | os.PathLike[bytes], file_name: str) -> Self:
        with open(f'{in_path}/{file_name}.json', 'r') as file:
            return self._from_dict(json.load(file))

    def remove(self, data_path: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> None:
        file_path = os.path.join(data_path, f'user{self.user_id}_build{self.build}.csv')
        if os.path.exists(file_path):
            os.remove(file_path)


class GenotypeIterator:
    def __init__(self, genotype: Genotype):
        self.genotype = genotype
        self.iter = genotype.genotype.iterrows()

    def __next__(self):
        index, row = next(self.iter)
        return index, row['genotype']