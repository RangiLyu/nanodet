import os
from .rank_filter import rank_filter


@rank_filter
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
