import ctypes
from ctypes import c_char_p, c_void_p, c_int, c_bool, c_size_t, \
    pointer, POINTER, c_uint64
from .config import jit_cu_lib_path, jit_llvm_lib_path
from .utils import CLib

jit_cu_lib = CLib(jit_cu_lib_path)
jit_cu_lib.register(c_size_t, 'cuda_compile_program', c_char_p,
                    c_int, c_char_p, c_bool, c_bool, c_bool, c_bool, c_char_p)
jit_cu_lib.register(c_void_p, 'cuda_load_module', c_void_p, c_char_p)
jit_cu_lib.register(POINTER(c_size_t), 'cuda_launch_kernel',
                    c_void_p, c_void_p, c_size_t, POINTER(c_void_p), c_void_p)
jit_cu_lib.register(c_void_p, 'cuda_get_kernel', c_void_p, c_void_p, c_char_p)
jit_cu_lib.register(None, 'cuda_unload_module', c_void_p, c_void_p)

jit_llvm_lib = CLib(jit_llvm_lib_path)
jit_llvm_lib.register(c_int, 'cpp_compile_program', c_char_p, c_char_p, c_char_p)
jit_llvm_lib.register(c_int, 'load_obj', c_char_p, c_char_p, c_char_p)
jit_llvm_lib.register(c_uint64, 'lookup', c_char_p, c_char_p)
jit_llvm_lib.register(c_int, 'unload_obj', c_char_p)
