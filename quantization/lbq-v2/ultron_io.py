import torch
import os

class UltronIO(object):
    def __init__(self, host, **pyarrow_args):
        pass

    def create_folder(self, folder_path, **mkdir_args):
        os.makedirs(folder_path, **mkdir_args)

    def check_path(self, file_path):
        return os.path.exists(file_path)

    def torch_load(self, file_path, **load_args):
        return torch.load(file_path, **load_args)

    def torch_save(self, file_path, obj, **save_args):
        torch.save(obj, file_path, **save_args)

