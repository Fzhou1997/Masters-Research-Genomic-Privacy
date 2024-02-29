import os
import warnings
import pandas as pd
from snps import SNPs

# We set the following to None due to the answer given in this
# StackOverflow post: https://stackoverflow.com/a/20627316
# TLDR: for us, this warning is a false positive.
pd.options.mode.chained_assignment = None  # default='warn'

# SNPs has a lot of DtypeWarnings unfortunately
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

HAIR_COLORS = {
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
"""The map of raw hair colors to parsed hair color."""


def parse_hair_color(hair_color: str) -> str:
    """Convert the raw hair color into one of: [`'blonde'`, `'brown'`, `'black'`]

    Args:
        hair_color (str): the raw hair color string.
    Returns:
        str: the parsed hair color.
    """
    return HAIR_COLORS.get(hair_color.lower(), '<drop>')


class PhenotypeLoader:
    """Utility loader class for phenotype CSV files.

    This class is used to connect hair color phenotypes to user genotype files.

    Usage:
    ```python
    >>> loader = PhenotypeLoader(RAW_PATH, ID)
    >>> print(loader.df)
    >>> print(loader.get_filename(USER_ID))
    ```
    """

    def __init__(self, raw_path: str, id: int, verbose=False):
        path = os.path.join(raw_path, f'phenotypes_{id}.csv')
        self.df = pd.read_csv(
            path,
            sep=';',
            na_values=['-', 'rather not say'],
            usecols=[
                'user_id',
                'genotype_filename',
                'date_of_birth',
                'chrom_sex',
                'Hair Color'
            ],
            dtype={
                'user_id': int,
                'genotype_filename': str,
                'date_of_birth': str,
                'chrom_sex': str,
                'Hair Color': str
            }
        )

        self._process_user_id()
        self._process_hair_color()
        self._process_date_of_birth()
        self._process_chrom_sex()
        self._process_file_id_and_provider()

        if verbose:
            print(f'Loaded {path}:')
            print(self.df)

    def _process_user_id(self):
        """Drop file IDs with duplicate user IDs (keep most recent file), and
        set user ID to be the index."""
        self.df.drop_duplicates('user_id', keep='last', inplace=True)
        self.df.set_index('user_id', inplace=True)

    def _process_hair_color(self):
        """Parse raw hair color to create strict labels."""
        self.df.rename(
            {'Hair Color': 'hair_color'},
            axis='columns', inplace=True
        )
        self.df.dropna(subset=['hair_color'], axis='index', inplace=True)
        self.df['hair_color'] = self.df['hair_color'].apply(parse_hair_color)
        self.df = self.df[self.df['hair_color'] != '<drop>']

    def _process_date_of_birth(self):
        """Replace NaN date of births with 'unknown'."""
        self.df['date_of_birth'] = self.df['date_of_birth'].fillna('unknown')

    def _process_chrom_sex(self):
        """Replace NaN chromosome sexes with 'unknown'."""
        self.df['chrom_sex'] = self.df['chrom_sex'].fillna('unknown')

    def _process_file_id_and_provider(self):
        """Extract file ID from genotype filename."""
        self.df[['provider', 'file_id']] = [
            tokens.split('.')[1:]  # (user_id, provider, file_id)
            for tokens in self.df['genotype_filename']
        ]
        self.df.drop('genotype_filename', axis='columns', inplace=True)

    def get_filename(self, user_id: int) -> str:
        """Get the raw genotype filename for a given user.

        Args:
            user_id (int): the user ID.
        Returns:
            str: the name of the file associated with that user.
        """
        row = self.df.loc[user_id]
        file_id = row['file_id']
        date_of_birth = row['date_of_birth']
        chrom_sex = row['chrom_sex']
        provider = row['provider']
        return f'user{user_id}_file{file_id}_yearofbirth_{date_of_birth}_sex_{chrom_sex}.{provider}.txt'


class SNPSLoader:
    """Utility loader class for raw genotype files.

    This class is used to parse the SNPs from genotype files.

    Optional Args:
        res_path (str): path where the snps library keeps resource files (like
        human assembly builds).

    Usage:
    ```python
    >>> loader = SNPSLoader(RAW_PATH, res_path=RESOURCES_PATH)
    >>> snps_df, build = loader.load(FILENAME)
    >>> snps_df_alleles, build = loader.load(FILENAME, expand_alleles=True)
    ```
    """

    def __init__(self, raw_path: str, res_path='resources'):
        self._raw_path = raw_path
        self._res_path = res_path

    def load(self, filename: str, expand_alleles=False):
        """Load the SNPS in a genotype sample from a given filename.

        Args:
            filename (str): the filename of the genotype.
            expand_alleles (bool, optional): whether to isolate alleles from their genotype.

        Returns:
            (DataFrame, int): a dataframe of the SNPs and the assembly build number, or None
            if the genotype could not be loaded.
        """
        path = os.path.join(self._raw_path, filename)
        try:
            s = SNPs(path, resources_dir=self._res_path)
            if not s.valid or not s.build_detected:
                return None
            df = s.snps
            df = df[df.chrom.isin(map(str, range(1, 23)))]
            df.fillna('--', inplace=True)
            if expand_alleles:
                for letter in ['A', 'C', 'G', 'T', 'D', 'I', '-']:
                    df[letter] = df['genotype']\
                        .apply(lambda genotype: genotype.count(letter))
            return df, s.build
        except:
            return None
