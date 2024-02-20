import warnings

import pandas as pd

from genotype_autosomal import process_users_txts
from phenotype_hair_color import load_raw, load_processed, save

if __name__ == '__main__':
    pd.options.mode.copy_on_write = True
    warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
    phenotype_dataframe = load_processed('../data/opensnp/phenotype/processed')
    process_users_txts('../data/opensnp/genotype/txt', '../data/opensnp/genotype/csv', phenotype_dataframe['user_id'].unique())
