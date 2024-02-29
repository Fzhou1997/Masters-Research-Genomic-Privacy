import os
import re
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

from proc.loaders import SNPSLoader


class Distribution:
    def __init__(self, filenames: list[str], loader=SNPSLoader):
        pass
