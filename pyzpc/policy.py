import os
from functools import partial
from .types import Module
from .config import clang_path
from .jit import jit_cu_lib, jit_llvm_lib
from .zpc import zpc_lib
from ctypes import c_int, c_float, c_double, c_size_t, c_void_p, c_char_p, addressof, CFUNCTYPE
from .types import Kernel, get_module
from .containers import MemSrc, TileVectorAPI, Container, TileVectorNamedViewType, TileVectorViewType


class Policy:
    def __init__(self) -> None:
        self.ptr = zpc_lib.lib.policy__device()

    def __del__(self):
        zpc_lib.lib.del_policy__device(self.ptr)


ctypes_lut = {
    int: c_int,
    float: c_float
}
# TODO: handle double and other types in a better way
# TODO: support zs small string


def args_to_ctypes(args):
    ret = []
    for arg in args:
        if hasattr(arg, 'ptr'):
            ret.append(c_void_p(arg.ptr))
            continue
        ret.append(ctypes_lut[type(arg)](arg))
    return ret


def launch(pol, py_kernel: Kernel, dims, *args, mode='cuda'):
    module: Module = py_kernel.module
    if mode == 'cuda':
        module.build_cuda()
    else:
        module.build_llvm()

    args = list(args)
    # TODO: make them properties of py_kernels; cache them when building modules
    args_types = py_kernel.args.annotations
    args_names = py_kernel.args.names

    # container -> view
    view_args = []
    for ind, arg in enumerate(args):
        if isinstance(arg, TileVectorAPI):
            arg_type = args_types[args_names[ind]]
            if isinstance(arg_type, TileVectorViewType):
                view_args.append(arg.view())
            elif isinstance(arg_type, TileVectorNamedViewType):
                view_args.append(arg.named_view())
            else:
                raise RuntimeError(
                    f'tv view type should be annotated, expected: TileVectorViewType \
                    or TileVectorNamedViewType, got: {type(arg_type)}')
        elif isinstance(arg, Container):
            view_args.append(arg.view())
        else:
            view_args.append(arg)

    # process prop tags for tv unnamed view
    for tvv_name in py_kernel.args.tv_unnamed_list:
        tv = args[py_kernel.args.inds[tvv_name]]
        for prop_tag in py_kernel.args.tv_unnamed_tags[tvv_name]:
            view_args.append(tv.tag_offset(prop_tag))

    # args -> ctype ptrs
    llvm_args = []
    arg_ptrs = []
    c_args = []
    for arg in view_args:
        if hasattr(arg, 'ptr'):
            arg_ptrs.append(c_void_p(arg.ptr))
            llvm_args.append(c_void_p(arg.ptr))
            continue
        ctypes_t = ctypes_lut.get(type(arg))
        c_arg = ctypes_lut[type(arg)](arg) if ctypes_t is not None else arg
        c_args.append(c_arg)
        arg_ptrs.append(c_void_p(addressof(c_arg)))
        llvm_args.append(c_arg)

    if mode == 'cuda':
        cu_module = jit_cu_lib.lib.cuda_load_module(
            pol.ptr, c_char_p(os.path.abspath(module.cuda_build_path).encode()))
        kernel = jit_cu_lib.lib.cuda_get_kernel(
            pol.ptr, cu_module, c_char_p(py_kernel.name.encode()))
        # launch kernel
        zpc_lib.lib.launch__device(pol.ptr, kernel, c_size_t(
            dims), (c_void_p * len(view_args))(*arg_ptrs))
        # TODO: check .encode() for all the c_char_p
    else:
        jit_llvm_lib.lib.load_obj(c_char_p(clang_path.encode()),
                                  c_char_p(os.path.abspath(
                                      module.llvm_build_path).encode()),
                                  c_char_p(module.name.encode()))
        llvm_func_ptr = jit_llvm_lib.lib.lookup(c_char_p(module.name.encode()),
                                                c_char_p(py_kernel.llvm_launch_symbol_name.encode()))
        llvm_func = CFUNCTYPE(None)(llvm_func_ptr)
        # print(llvm_func)
        llvm_func(*llvm_args, c_size_t(dims))
        jit_llvm_lib.lib.unload_obj(c_char_p(module.name.encode()))


launch_cuda = partial(launch, mode='cuda')
launch_llvm = partial(launch, mode='llvm')
