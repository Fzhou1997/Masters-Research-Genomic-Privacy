import pandas as pd

genotype1 = pd.DataFrame()
genotype1.index.name = 'rsid'
genotype1['genotype'] = None
genotype1.loc['rs0'] = ['AA']

genotype2 = pd.DataFrame()
genotype2.index.name = 'rsid'
genotype2['genotype'] = None
genotype2.loc['rs1'] = ['CC']

genotype3 = pd.DataFrame()
genotype3.index.name = 'rsid'
genotype3['genotype'] = None
genotype3.loc['rs0'] = ['GG']
genotype3.loc['rs1'] = ['TT']

genotype4 = pd.DataFrame()
genotype4.index.name = "rsid"

genomes = pd.DataFrame()
user1 = genotype1.T
user1.index = pd.Index([1], name='user_id')
genomes = pd.concat([genomes, user1], ignore_index=False)
user2 = genotype2.T
user2.index = pd.Index([2], name='user_id')
genomes = pd.concat([genomes, user2], ignore_index=False)
user3 = genotype3.T
user3.index = pd.Index([3], name='user_id')
genomes = pd.concat([genomes, user3], ignore_index=False)
print(genomes)

# df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['2', '$F', '5'])
# df_copy = df.copy()
# df_copy.index = df_copy.index.str.lower()
# df_copy.index = df_copy.index.str.replace(r'[^0-9a-z]+', '', regex=True)
# print(df_copy)