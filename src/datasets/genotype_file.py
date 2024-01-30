
def find_between(s, start, end):
    return (s.split(start))[1].split(end)[0]


class GenotypeFile:
    def __init__(self, file_name, version, data):
        self.file_name = file_name  # str
        self.version = version  # int
        self.data = data  # pandas dataframe

        self.user_id = int(find_between(file_name, 'user', '_file'))
        self.file_id = int(find_between(file_name, '_file', '_yearofbirth_'))
        self.year_of_birth = find_between(file_name, "_yearofbirth_", '_sex_')
        self.chromosome_sex = find_between(file_name, '_sex_', '.')
        self.provider = file_name.split('.')[-2]


