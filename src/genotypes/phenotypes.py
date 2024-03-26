import os

import pandas as pd


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


class Phenotypes:
    def __init__(self):
        self.phenotypes = None
        self.feature_name = ''

    def from_feature(self, data_path, feature_name, converter=None):
        phenotypes_file_name = ''
        for file_name in os.listdir(data_path):
            if file_name.startswith('phenotypes_') and file_name.endswith('.csv'):
                phenotypes_file_name = file_name
                break
        if not phenotypes_file_name:
            raise FileNotFoundError('No phenotypes file found')
        self.feature_name = re.sub(r'\s+', '_', re.sub(r'[^a-zA-Z0-9_\s]+', ' ', feature_name.strip().lower()))
        self.phenotypes = pd.read_csv(os.path.join(data_path, phenotypes_file_name),
                                      delimiter=';',
                                      na_values=['-', 'rather not say'],
                                      converters={feature_name: converter} if converter else None,
                                      usecols=['user_id'] + [feature_name],
                                      index_col=0)
        self.phenotypes.columns = self.phenotypes.columns.str.replace(r'[^a-zA-Z0-9_\s]+', ' ', regex=True).str.strip().str.lower().str.replace(r'\s+', '_', regex=True)
        self.phenotypes = self.phenotypes.dropna()
        self.phenotypes = self.phenotypes.groupby(self.phenotypes.index).last()
        self.phenotypes = self.phenotypes.sort_index()

    def is_encoded_one_hot(self):
        return self.phenotypes.shape[1] > 1

    def get_phenotypes(self):
        return self.phenotypes.copy()

    def get_user_ids(self):
        return self.phenotypes.index

    def get_values(self):
        return list(self.phenotypes[self.feature_name].unique())

    def get_one_hot(self):
        return pd.get_dummies(self.phenotypes[self.feature_name], prefix=self.feature_name)

    def save(self, out_path):
        out = self.phenotypes.reset_index()
        os.makedirs(out_path, exist_ok=True)
        out.to_csv(os.path.join(out_path, f'phenotypes_{self.feature_name}.csv'), index=False)

    def load(self, data_path, feature_name):
        file_path = os.path.join(data_path, f'phenotypes_{feature_name}.csv')
        self.feature_name = feature_name
        self.phenotypes = pd.read_csv(file_path, index_col=0)
