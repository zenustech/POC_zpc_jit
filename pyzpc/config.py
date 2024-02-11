import os
import sys

base_path = os.path.dirname(__file__)
lib_path = os.path.join(base_path, 'zpc_jit', 'lib')

gen_cu_path = os.path.join(base_path, '.cache', 'gen', 'cu')
gen_cu_hash_path = os.path.join(gen_cu_path, 'hash')
gen_cu_args_path = os.path.join(gen_cu_path, 'args')
gen_llvm_path = os.path.join(base_path, '.cache', 'gen', 'llvm')
gen_llvm_hash_path = os.path.join(gen_llvm_path, 'hash')
gen_llvm_args_path = os.path.join(gen_llvm_path, 'args')
llvm_obj_suffix = 'obj' if os.name == 'nt' else 'o'
lib_suffix = 'so'
if sys.platform == 'win32':
    lib_suffix = 'dll'
elif sys.platform == 'darwin':
    lib_suffix = 'dylib'
lib_prefix = '' if os.name == 'nt' else 'lib'
jit_cu_lib_path = os.path.join(lib_path, f'{lib_prefix}zpc_jit_nvrtc.{lib_suffix}')
jit_llvm_lib_path = os.path.join(lib_path, f'{lib_prefix}zpc_jit_clang.{lib_suffix}')
clang_path = f'libclang.{lib_suffix}'
has_omp_lib = os.path.exists(os.path.join(lib_path, f'libomp.{lib_suffix}'))

zpc_lib_path = os.path.join(lib_path, f'{lib_prefix}zpc_py_interop.{lib_suffix}')
zpc_include_path = os.path.join(base_path, 'zpc_jit', 'zpc', 'include')


debug_print = False
