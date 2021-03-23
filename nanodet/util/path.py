import os
from .rank_filter import rank_filter


@rank_filter
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        assert not os.path.exists(path), f"{path} already exists, in order not to overwrite this folder, please modify the save_dir in config yaml or delete this folder manually"
