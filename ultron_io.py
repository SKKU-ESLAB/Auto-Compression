import io
import torch
import multiprocessing as mp
import os
import pyarrow as pa


def hdfs_create_folder(host, pyarrow_args, folder_path, mkdir_args):
    hdfs = pa.hdfs.connect(host, **pyarrow_args)
    hdfs.mkdir(folder_path, **mkdir_args)


def hdfs_torch_load(host, pyarrow_args, file_path, load_args):
    hdfs = pa.hdfs.connect(host, **pyarrow_args)
    with hdfs.open(file_path, mode='rb') as f:
        return torch.load(io.BytesIO(f.read()), **load_args)


def hdfs_torch_save(host, pyarrow_args, file_path, obj, save_args):
    hdfs = pa.hdfs.connect(host, **pyarrow_args)
    with hdfs.open(file_path, mode='wb') as f:
        buffer = io.BytesIO()
        torch.save(obj, buffer, **save_args)
        f.write(buffer.getvalue())

def hdfs_check_path(host, pyarrow_args, path):
    hdfs = pa.hdfs.connect(host, **pyarrow_args)
    return hdfs.exists(path)

def is_hdfs_path(path):
    # TODO This is a hack!!! Theoretically hdfs path can be legal path
    # string.
    return path.startswith('hdfs://') or path.startswith('/home')


class UltronIO(object):
    def __init__(self, host, **pyarrow_args):
        self.host = host
        self.pyarrow_args = pyarrow_args
        self.worker = mp.Pool(1)

    def create_folder(self, folder_path, **mkdir_args):
        if is_hdfs_path(folder_path):
            self.worker.apply(
                hdfs_create_folder,
                (self.host, self.pyarrow_args, folder_path, mkdir_args))
        else:
            os.makedirs(folder_path, **mkdir_args)
    def check_path(self, file_path):
        if is_hdfs_path(file_path): 
          return self.worker.apply(
              hdfs_check_path,
              (self.host, self.pyarrow_args, file_path))
        else:
          return os.path.exists(file_path)

    def torch_load(self, file_path, **load_args):
        if is_hdfs_path(file_path):
            return hdfs_torch_load(self.host, self.pyarrow_args,
                file_path, load_args)
            #return self.worker.apply(
            #    hdfs_torch_load,
            #    (self.host, self.pyarrow_args, file_path, load_args))
        else:
            return torch.load(file_path, **load_args)

    def torch_save(self, file_path, obj, **save_args):
        if is_hdfs_path(file_path):
            return hdfs_torch_save(self.host, self.pyarrow_args,
                file_path, obj, save_args)
            #self.worker.apply(
            #    hdfs_torch_save,
            #    (self.host, self.pyarrow_args, file_path, obj, save_args))
        else:
            torch.save(obj, file_path, **save_args)
    def __del__(self):
      self.worker.close()
