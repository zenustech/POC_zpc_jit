import os
import ctypes
import hashlib

if os.name == 'nt':
    os.add_dll_directory('C:\\Windows\\System32\\downlevel')
    if 'CUDA_PATH' in os.environ:
        os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))


def check_folder(path):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def hash_update_int(h, integer):
    while integer > 0:
        new_val = integer >> 128
        cur_128 = integer - (new_val << 128)
        h.update(cur_128.to_bytes(16, 'little', signed=False))
        integer = new_val


class CLib:
    def __init__(self, lib_path=None) -> None:
        self.lib_path = lib_path
        if lib_path is not None:
            self.init_lib(lib_path)

    def init_lib(self, lib_path):
        if os.path.exists(lib_path):
            self.lib = ctypes.cdll.LoadLibrary(lib_path)
        else:
            self.lib = None

    def register(self, restype, func_name, *argtypes):
        if self.lib is None:
            print(f'Library {self.lib_path} is not found, \
                  please check if CUDA and LLVM are installed properly.')
        func = getattr(self.lib, func_name)
        func.restype = restype
        func.argtypes = argtypes

    def call(self, func_name, *args):
        func = getattr(self.lib, func_name)
        return func(*args)


str2ctype = {
    'int': ctypes.c_int,
    'double': ctypes.c_double,
    'float': ctypes.c_float
}
ctype2str = {
    v: k for k, v in str2ctype.items()
}
