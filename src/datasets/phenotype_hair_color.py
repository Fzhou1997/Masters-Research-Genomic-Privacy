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


def preprocess_hair_colors(string):
    if string in hair_color_dict:
        return hair_color_dict[string]
    else:
        return ''


class PhenotypeHairColor:
    def __init__(self,
                 load_path,
                 save_path,
                 delimiter=None,
                 na_values=None,
                 unsupported_providers=None,
                 hair_color_dict=None):
        self.load_path = load_path
        self.save_path = save_path
        self.delimiter = delimiter
        self.na_values = na_values
        self.unsupported_providers = unsupported_providers
        self.hair_color_dict = hair_color_dict

    def load(self, file_name):
        file_path = os.path.join(self.load_path, file_name)
        dataframe = pd.read_csv(file_path,
                                delimiter=self.delimiter,
                                na_values=self.na_values,
                                usecols=['user_id', 'Hair Color'],
                                converters={'Hair Color': preprocess_hair_colors})
        dataframe = dataframe.rename(columns={'Hair Color': 'hair_color'})
        dataframe = dataframe[~dataframe['hair_color'].isin(self.unsupported_providers)]
        dataframe = dataframe.dropna()
        return dataframe

    def save(self, dataframe, file_name):
        file_path = os.path.join(self.save_path, file_name)
        dataframe.to_csv(file_path, index=False)
        return file_path

