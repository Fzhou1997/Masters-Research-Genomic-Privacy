import numpy as np
import scipy.sparse as sp

from utils_snp import SnpMap



class SnpCounter:
    def __init__(self, snp_map: SnpMap):
        self.bld = snp_map.get_build()
        self.chr = snp_map.get_chromosome()
        self.snps = snp_map.get_snps()
        self.cnts = np.zeros((len(snp_map),), dtype=int, )

