from .decorators import func, kernel, llvm_func, llvm_kernel, cuda_func, cuda_kernel
from .types import get_module  # for debugging purposes
from .policy import Policy, launch, MemSrc, launch_cuda, launch_llvm
from .policy import ctypes_lut
# from .jit import jit_cu_lib
from .jit import init_zpc_clang_lib
from .jit import init_zpc_nvrtc_lib
from .zpc import init_zpc_lib
from .zeno import init_zeno_lib

import os
from .config import zpc_lib_path 
if zpc_lib_path is not None:
    if os.path.exists(zpc_lib_path):
        init_zpc_lib(zpc_lib_path)
    else:
        print('default pyzpc capi lib path [{zpc_lib_path}] does not exist')
else:
    print('pyzpc capi lib path not specified, require user\' explicit [init_zpc_lib(path)]')

from .config import jit_cu_lib_path  
if jit_cu_lib_path is not None:
    if os.path.exists(jit_cu_lib_path):
        init_zpc_nvrtc_lib(jit_cu_lib_path)
    else:
        print('default zpcjit nvrtc lib path [{jit_cu_lib_path}] does not exist')
else:
    print('zpcjit nvrtc lib path not specified, require user\' explicit [init_zpc_nvrtc_lib(path)]')

from .config import jit_llvm_lib_path  
if jit_llvm_lib_path is not None:
    if os.path.exists(jit_llvm_lib_path):
        init_zpc_clang_lib(jit_llvm_lib_path)
    else:
        print('default zpcjit clang lib path [{jit_llvm_lib_path}] does not exist')
else:
    print('zpcjit clang lib path not specified, require user\' explicit [init_zpc_clang_lib(path)]')


from .containers import TileVector, f32, f64, i32, i64, fl, dbl, \
    double, string, tvv_t, tvnv_t, tv_t, vv_t, v_t, integer, SmallVec, \
    svec, svec_t, SmallVecObject, VectorObject, TileVectorObject, vec2i, \
    vec3i, vec4i, vec2f, vec3f, vec4f, mat2i, mat3i, mat4i, mat2f, mat3f, \
    mat4f, Vector

float = fl
int = integer
tv = TileVector

# TODO: test SmallVecObject; then add typing alias
__all__ = ['func', 'kernel', 'f32', 'f64', 'i32',
           'i64', 'fl', 'dbl', 'double', 'string',
           'tvv_t', 'tvnv_t', 'tv_t', 'get_module',
           'Policy', 'launch', 'tv', 'vv_t', 'v_t',
           'jit_cu_lib', 'zpc_lib', 'ctypes_lut', 'int',
           'i', 'svec', 'svec_t', 'translate_sym_expr',
           'SmallVecObject', 'init_zeno_lib',
           'VectorObject', 'TileVectorObject',
           'launch_cuda', 'launch_llvm',
           'llvm_func', 'llvm_kernel',
           'cuda_func', 'cuda_kernel',
           'Vector', 'vec2i', 'vec3i', 'vec4i',
           'vec2f', 'vec3f', 'vec4f',
           'mat2i', 'mat3i', 'mat4i',
           'mat2f', 'mat3f', 'mat4f']
