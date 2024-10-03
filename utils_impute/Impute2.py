import os
from os import PathLike

from utils_io.windows_to_wsl_path import windows_to_wsl_path

class Impute2:
    def __init__(self,
                 impute2_path: str | bytes | PathLike[str] | PathLike[bytes],
                 use_wsl: bool = False):
        if not os.path.exists(impute2_path):
            raise FileNotFoundError(f"Impute2 not found at {impute2_path}")
        if not os.access(impute2_path, os.X_OK):
            raise PermissionError(f"Impute2 not executable at {impute2_path}")
        if not os.path.isfile(impute2_path):
            raise IsADirectoryError(f"Impute2 is a directory at {impute2_path}")
        impute2_path = os.path.abspath(impute2_path)
        self.command = ""
        if use_wsl:
            self.command += "wsl "
            impute2_path = windows_to_wsl_path(impute2_path)
        self.command += impute2_path
        self.args = []

    def impute(self,
               g: str | bytes | PathLike[str] | PathLike[bytes],
               m: str | bytes | PathLike[str] | PathLike[bytes],
               interval: tuple[int, int],
               h: str | bytes | PathLike[str] | PathLike[bytes] = None,
               l: str | bytes | PathLike[str] | PathLike[bytes] = None,
               g_ref: str | bytes | PathLike[str] | PathLike[bytes] = None,
               known_haps_g: str | bytes | PathLike[str] | PathLike[bytes] = None,
               o: str | bytes | PathLike[str] | PathLike[bytes] = None,
               i: str | bytes | PathLike[str] | PathLike[bytes] = None,
               r: str | bytes | PathLike[str] | PathLike[bytes] = None,
               w: str | bytes | PathLike[str] | PathLike[bytes] = None,
               osnps: list[int, int] = None,
               o_gz: bool = False,
               outdp: int = None,
               no_snp_qc_info: bool = False,
               no_sample_qc_info: bool = False,
               phase: bool = False,
               pgs: bool = False,
               pgs_miss: bool = False,
               buffer: int = None,
               allow_large_regions: bool = False,
               include_buffer_in_output: bool = False,
               Ne: int = None,
               call_thresh: float = None,
               nind: int = None,
               verbose: bool = False,
               strand_g: str | bytes | PathLike[str] | PathLike[bytes] = None,
               strand_g_ref: str | bytes | PathLike[str] | PathLike[bytes] = None,
               align_by_maf_g: bool = False,
               align_by_maf_g_ref: bool = False,
               filt_rules_l: list[str] = None,
               exclude_snps_g: str | bytes | PathLike[str] | PathLike[bytes] = None,
               iter: int = None,
               burnin: int = None,
               k: int = None,
               k_hap: int = None,
               prephase_g: bool = False,
               use_prephased_g: bool = False,
               merge_ref_panels: bool = False,
               merge_ref_panels_output_ref: str | bytes | PathLike[str] | PathLike[bytes] = None,
               merge_ref_panels_output_gen: str | bytes | PathLike[str] | PathLike[bytes] = None,
               chrX: bool = False,
               Xpar: bool = False,
               seed: int = None,
               no_warn: bool = False,
               fill_holes: bool = False,
               no_remove: bool = False):
        pass
