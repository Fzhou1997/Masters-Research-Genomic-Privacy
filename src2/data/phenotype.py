import os

import pandas as pd
from pandas import DataFrame

hair_color_dict = {
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


def convert_hair_colors(string):
    string = string.lower()
    if string in hair_color_dict:
        return hair_color_dict[string]
    else:
        return None


class PhenotypeProcessor:
    def __init__(self, raw_path: str, out_path: str):
        self.raw_path = raw_path
        self.out_path = out_path
        self.delimiter = ';'
        self.na_values = ['-', 'rather not say']
        self.usecols = ['user_id', 'Hair Color']
        self.converters = {'Hair Color': convert_hair_colors}
        self.phenotype_fn = ""
        for filename in os.listdir(self.raw_path):
            if filename.startswith('phenotypes_') and filename.endswith('.csv'):
                self.phenotype_fn = filename
                break
        self.phenotype_fp = os.path.join(self.raw_path, self.phenotype_fn)


    def set_user_ids(self, user_ids: set[int]):
        """Sets the user IDs to filter the phenotype data."""
        self.user_ids = user_ids

    def load_raw(self) -> DataFrame:
        """Loads the raw phenotype data."""
        phenotype_df = pd.read_csv(self.phenotype_fp,
                                   delimiter=self.delimiter,
                                   na_values=self.na_values,
                                   usecols=self.usecols,
                                   converters=self.converters)
        phenotype_df = phenotype_df.rename(columns={'Hair Color': 'hair_color'})
        phenotype_df = phenotype_df[~phenotype_df.apply(lambda row: row.str.contains('exome-vcf|IYG').any(), axis=1)]
        phenotype_df = phenotype_df.dropna()
        phenotype_df = phenotype_df.drop_duplicates()
        phenotype_df = phenotype_df.sort_values(by='user_id')
        phenotype_df = phenotype_df.reset_index(drop=True)
        return phenotype_df

    def load_processed(self) -> DataFrame:
        """Loads the processed phenotype data."""
        phenotype_filepath = os.path.join(self.out_path, 'phenotype_hair_color.csv')
        phenotype_dataframe = pd.read_csv(phenotype_filepath)
        return phenotype_dataframe

    def save_processed(self, phenotype_df: DataFrame):
        """Saves the processed phenotype data."""
        phenotype_filepath = os.path.join(self.out_path, 'phenotype_hair_color.csv')
        phenotype_df.to_csv(phenotype_filepath, index=False)

    def filter(self, phenotype_df: DataFrame) -> DataFrame:
        """Filters the phenotype data for a given set of user IDs."""
        phenotype_df = phenotype_df[phenotype_df['user_id'].isin(self.user_ids)]
        phenotype_df = phenotype_df.sort_values(by='user_id')
        phenotype_df = phenotype_df.reset_index(drop=True)
        return phenotype_df
