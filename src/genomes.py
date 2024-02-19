import os
import re

import pandas as pd


def load_raw(data_path):
    files = os.listdir(data_path)
    data = []
    for file_name in files:
        file_extension = file_name.split('.')[-1]
        if file_extension != 'txt':
            continue
        match = re.search(r"user(\d+)_file(\d+)_yearofbirth_(.+)_sex_(.+)\.(.+)\.txt", file_name)
        if match:
            user_id = int(match.group(1))
            file_id = int(match.group(2))
            year_of_birth = match.group(3)
            sex = match.group(4)
            provider = match.group(5)
            data.append([user_id, file_id, year_of_birth, sex, provider])
    dataframe = pd.DataFrame(data, columns=['user_id', 'file_id', 'year_of_birth', 'sex', 'provider'])
    dataframe = dataframe.sort_values(by=['user_id', 'file_id'])
    dataframe = dataframe.reset_index(drop=True)
    return dataframe


def load_processed(data_path):
    dataframe = pd.read_csv(os.path.join(data_path, 'genomes.csv'))
    return dataframe


def save(genomes, file_path):
    genomes.to_csv(file_path, index=False)

