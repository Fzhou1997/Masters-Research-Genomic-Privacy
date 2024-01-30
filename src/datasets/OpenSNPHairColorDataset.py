import os

import numpy as np
import pandas as pd

# Path variables
OPEN_SNP_DIR = "./../../datasets/OpenSNP/"
CSV_FILENAME = "phenotypes.csv"

# Load in the dataset
df = pd.read_csv(os.path.join(OPEN_SNP_DIR, CSV_FILENAME),
                 delimiter=';', na_values=['-','rather not say'], low_memory=False)

# Clean column names
df.columns = map(str.lower, df.columns)
df.columns = map(lambda column: column.replace(' ', '_'), df.columns)

# Combine duplicate columns into one
combined_columns = df['hair_color'].bfill(axis=1).iloc[:, 0]
df = df.drop('hair_color', axis=1)
df['hair_color'] = combined_columns

# Lowercase hair colors
df['hair_color'] = df['hair_color'].str.lower()

# Drop rows with missing hair colors
df.dropna(subset=['hair_color'], inplace=True)

# Drop specific files that are too few to worry about
df = df[df.apply(lambda row: 'exome-vcf' not in row['genotype_filename'], axis=1)]
df = df[df.apply(lambda row: 'IYG' not in row['genotype_filename'], axis=1)]

# Remove extra information the dataset
df = df[['user_id', 'hair_color']]
df = df.dropna()



print(df.head())
print(df.shape)
print(len(df["user_id"].unique()))
print(df["hair_color"].unique())