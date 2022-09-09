import os
import torch

def drop_file_type(filename: str, filetype: str):
    if filename.endswith("." + filetype):
        filename_ = filename.split("." + filetype)[0]
    else:
        filename_ = filename
    return filename_


def make_dirs(add):
    if os.path.exists(add):
        os.system(f"rm -r {add}")
    os.makedirs(add)

def get_torch_device()-> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")