import json
from os import PathLike, makedirs, listdir
import re
from typing import Self

import pandas as pd

HAIR_COLOR_ENCODER_READABLE = {
    ' light blonde as a child and medium blonde as an adult. ': 'blonde',
    'auburn': 'brown',
    'auburn (reddish-brown)': 'brown',
    'black': 'black',
    'black ': 'black',
    'black (very slight tint of red)': 'black',
    'blackish brown': 'brown',
    'blond': 'blonde',
    'blond as a child and light brown as an adult': 'brown',
    'blond as a child.  dark blond as an adult.': 'blonde',
    'blond as child. started turning dark brown after puberty': 'brown',
    'blond born, today dark brown': 'brown',
    'blonde': 'blonde',
    'blonde as a child, light brown as an adult': 'brown',
    'blonde as a child, to brown as an adult': 'brown',
    'blonde as child, ash blonde as adult, early white': 'blonde',
    'blonde to light brown as child, medium brown as adult with blonde highlights from sun': 'brown',
    'blondish reddish brown': 'brown',
    'bright copper ginger into my 40s. light auburn with grey temples as i age.': 'brown',
    'brown': 'brown',
    'brown and silver': 'brown',
    'brown going to white in early 40s': 'brown',
    'brown,red,blond': 'brown',
    'brown-black': 'black',
    'chestnut brown': 'brown',
    'copper/red': 'brown',
    'dark blonde': 'blonde',
    'dark blonde ': 'blonde',
    'dark blonde (light brown)': 'brown',
    'dark blonde as a child, chestnut brown as an adult': 'brown',
    'dark blonde as a child, dark brown as an adult': 'brown',
    'dark blonde with a little of every colour but black.': 'blonde',
    'dark blonde, ': 'blonde',
    'dark blonde, strawberry': 'blonde',
    'dark brown': 'brown',
    'dark brown; blonde highlights': 'brown',
    'dark brown; red highlights': 'brown',
    'darkest brown to black': 'black',
    'darkest brown to black ': 'black',
    'dirt-blonde': 'blonde',
    'dirt-brown': 'brown',
    'dirty blond, dark red beard': 'blonde',
    'dirty blonde, light brown, something?': 'brown',
    'grey and brown': 'brown',
    'grey head, strawberry blonde facial and body': 'blonde',
    'hair darkening with age, starting blonde, ending dark brown': 'brown',
    'light ashy brown': 'brown',
    'light brown': 'brown',
    'light to medium brown': 'brown',
    'medium brown': 'brown',
    'medium brown with highlights': 'brown',
    'medium brown, red highlights': 'brown',
    'medium golden brown': 'brown',
    'red': 'brown',
    'red (gone blond-grey)': 'brown',
    'reddish brown': 'brown',
    'reddish-brown': 'brown',
    'strawberry blond as a child, now dark auburn brown': 'brown',
    'strawberry blonde': 'blonde',
    'strawberry brown': 'brown',
    'toe head to dark reddish brown': 'brown',
    'towhead to light ashy brown by 20s': 'brown',
    'very dark brown': 'brown',
}

HAIR_COLOR_ENCODER_ORDINAL = {
    'blonde': 0,
    'brown': 1,
    'black': 2,
}


class Phenotype:
    def __init__(self):
        self.feature = ''
        self.phenotypes = None

    def __iter__(self):
        return PhenotypeIterator(self)

    def __getitem__(self, idx: int) -> str | int | None:
        if idx not in self.phenotypes.index:
            return None
        else:
            return self.phenotypes.at[idx, self.feature]

    def _to_dict(self) -> dict[str, str | dict[int, dict[str, str]]]:
        return {
            'feature': self.feature,
            'phenotypes': self.phenotypes.to_dict(orient='index'),
        }

    def _from_dict(self, data: dict[str, str | dict[int, dict[str, str]]]) -> Self:
        self.feature = data['feature']
        self.phenotypes = pd.DataFrame.from_dict(data['phenotypes'], orient='index')
        self.phenotypes.index.name = 'user_id'
        return self

    def from_feature(self, in_path: str | bytes | PathLike[str] | PathLike[bytes], feature: str) -> Self:
        file_path = None
        for file_name in listdir(in_path):
            if file_name.startswith(f'phenotypes_') and file_name.endswith('.csv'):
                file_path = f"{in_path}/{file_name}"
        if file_path is None:
            raise FileNotFoundError('No phenotype files found')
        self.feature = re.sub(r'\s+', '_', re.sub(r'[^a-zA-Z0-9_\s]+', ' ', feature.strip().lower()))
        self.phenotypes = pd.read_csv(file_path, delimiter=';', index_col=0)
        self.phenotypes.index.name = 'user_id'
        self.phenotypes.columns = self.phenotypes.columns.str.replace(r'[^a-zA-Z0-9_\s]+', ' ', regex=True).str.strip().str.lower().str.replace(r'\s+', '_', regex=True)
        self.phenotypes = self.phenotypes[self.feature]
        self.phenotypes = self.phenotypes.iloc[:, [0]]
        return self

    def clean(self) -> Self:
        self.phenotypes = self.phenotypes.groupby(self.phenotypes.index).last()
        self.phenotypes = self.phenotypes.sort_index()
        self.phenotypes = self.phenotypes.dropna()
        self.phenotypes = self.phenotypes[~self.phenotypes.index.duplicated(keep='first')]
        self.phenotypes.index = self.phenotypes.index.astype(int)
        self.phenotypes[self.feature] = self.phenotypes[self.feature].astype(str)
        self.phenotypes[self.feature] = self.phenotypes[self.feature].str.lower()
        if self.phenotypes.empty:
            raise ValueError('No valid phenotypes found')
        return self

    def encode(self, encoder: dict[str, str | int]) -> Self:
        self.phenotypes[self.feature] = self.phenotypes[self.feature].map(encoder, na_action='ignore')
        self.phenotypes = self.phenotypes.dropna()
        if self.phenotypes.empty:
            raise ValueError('No valid phenotypes found')
        return self

    def get_feature(self) -> str:
        return self.feature

    def get_phenotypes(self) -> pd.DataFrame:
        return self.phenotypes

    def get_user_ids(self) -> list[int]:
        return list(self.phenotypes.index)

    def get_values(self) -> list[str] | list[int]:
        return list(self.phenotypes[self.feature].unique())

    def get_one_hot(self) -> pd.DataFrame:
        return pd.get_dummies(self.phenotypes[self.feature], prefix=self.feature)

    def save(self, out_path: str | bytes | PathLike[str] | PathLike[bytes], file_name: str) -> None:
        makedirs(out_path, exist_ok=True)
        with open(f'{out_path}/{file_name}.json', 'w') as file:
            json.dump(self._to_dict(), file)

    def load(self, in_path: str | bytes | PathLike[str] | PathLike[bytes], file_name):
        with open(f'{in_path}/{file_name}.json', 'r') as file:
            return self._from_dict(json.load(file))


class PhenotypeIterator:
    def __init__(self, phenotype: Phenotype):
        self.phenotype = phenotype
        self.feature = phenotype.get_feature()
        self.iter = phenotype.phenotypes.iterrows()

    def __next__(self) -> tuple[int, str]:
        user_id, row = next(self.iter)
        return user_id, row[self.feature]
