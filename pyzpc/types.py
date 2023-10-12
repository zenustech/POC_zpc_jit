import os
import inspect
import hashlib
import pickle as pkl
from typing import Any
from .translators import FunctionTranslator, KernelTranslator
from .context import registered_modules
from .config import gen_cu_path, debug_print, zpc_include_path, \
    gen_llvm_path, llvm_obj_suffix, gen_cu_hash_path, gen_llvm_hash_path, \
    gen_cu_args_path, gen_llvm_args_path
from .utils import check_folder, hash_update_int
from .jit import jit_cu_lib, jit_llvm_lib
from ctypes import byref, pointer, c_char_p, c_int, c_bool
from .containers import DataType


class Function:
    def __init__(self, name, module, ret=None, func=None,
                 use_cuda: bool = True, use_llvm: bool = True):
        self.name = name
        self.ret = ret
        self.func = func
        self.module = module
        self.use_cuda = use_cuda
        self.use_llvm = use_llvm
        self.src = inspect.getsource(func)
        self.cuda_src = None
        self.llvm_src = None
        self.translator = None

    def update_hash(self, h):
        h.update(bytes(self.src, 'utf-8'))
        globals_hash = 0
        for type_name, zs_type in self.func.__globals__.items():
            if isinstance(zs_type, DataType) and zs_type.in_kernel:
                kv = str((type_name, zs_type.name)).encode()
                globals_hash += int(hashlib.sha256(kv).hexdigest(), 16)
        hash_update_int(h, globals_hash)

        for _, arg_type in self.func.__annotations__.items():
            h.update(arg_type.name.encode())

    def build(self):
        self.update_translator()
        if self.use_cuda:
            self.translator.translate(compile_mode='cuda')
            self.cuda_src = self.translator.cuda_header + \
                self.translator.cuda_src
        if self.use_llvm:
            self.translator.translate(compile_mode='llvm')
            self.llvm_src = self.translator.llvm_header + \
                self.translator.llvm_src
        self.args = self.translator.args_info

        if debug_print:
            self.debug_show_translation()

    def update_translator(self):
        self.translator = FunctionTranslator(self.func)

    def debug_show_translation(self):
        print(
            '-' * 10 + f"translation result of func {self.func}: " + '-' * 10)
        if self.use_cuda:
            print(self.cuda_src)
        print('-' * 20)
        if self.use_llvm:
            print(self.llvm_src)

    @property
    def call_name(self):
        return self.name


class Kernel(Function):
    def __init__(self, name, module, func=None,
                 use_cuda: bool = True, use_llvm: bool = True):
        super().__init__(name, module,
                         func=func,
                         use_cuda=use_cuda,
                         use_llvm=use_llvm)
        self.llvm_launch_symbol_name = f'__zs_gen_launch_{self.name}'

    def update_translator(self):
        self.translator = KernelTranslator(
            self.func, self.llvm_launch_symbol_name)

    def debug_show_translation(self):
        print(
            '-' * 10 + f"translation result of kernel {self.func}: " + '-' * 10)
        if self.use_cuda:
            print(self.cuda_src)
        print('-' * 20)
        if self.use_llvm:
            print(self.llvm_src)


# build_llvm = False is for debugging
class Module:
    def __init__(self, name, build_cuda=True, build_llvm=False) -> None:
        self.name = name
        self.funcs = []
        self.kernels = []
        self.sym_funcs = []
        self.translated_src = None

        self.cuda_out_args_path = os.path.join(
            gen_cu_args_path, f'{self.name}.pkl')
        self.llvm_out_args_path = os.path.join(
            gen_llvm_args_path, f'{self.name}.pkl')
        self.cuda_build_path = os.path.join(gen_cu_path, f'{self.name}.cubin')
        self.cuda_out_src_path = os.path.join(gen_cu_path, f'{self.name}.cu')
        self.cuda_hash_path = os.path.join(
            gen_cu_hash_path, f'{self.name}.hash')
        self.llvm_build_path = os.path.join(
            gen_llvm_path, f'{self.name}.{llvm_obj_suffix}')
        self.llvm_out_src_path = os.path.join(
            gen_llvm_path, f'{self.name}.cpp')
        self.llvm_hash_path = os.path.join(
            gen_llvm_hash_path, f'{self.name}.hash')

        if build_cuda:
            self.build_cuda()
        if build_llvm:
            self.build_llvm()

    def build_cuda(self):
        self.build(self.cuda_build_path,
                   self.cuda_out_src_path,
                   self.cuda_hash_path,
                   self.cuda_out_args_path,
                   'cuda')

    def build_llvm(self):
        self.build(self.llvm_build_path,
                   self.llvm_out_src_path,
                   self.llvm_hash_path,
                   self.llvm_out_args_path,
                   'llvm')

    def build(self, build_path, out_src_path, hash_path, args_path, compile_mode='cuda'):
        cache_hash = None
        if os.path.exists(hash_path):
            with open(hash_path, 'rb') as f:
                cache_hash = f.read()
        cur_hash = self.hash()
        if cache_hash == cur_hash:
            with open(args_path, 'rb') as f:
                args_cache = pkl.load(f)
            for func in self.funcs + self.kernels:
                func.args = args_cache[func.name]
            return
        translated_src = '#include "zensim/ZpcBuiltin.hpp"\n' + \
            '#include "zensim/py_interop/TileVectorView.hpp"\n\n'
        for f in self.funcs + self.sym_funcs:
            f.build()
            build_func = f.use_cuda if compile_mode == 'cuda' else f.use_llvm
            if build_func:
                translated_src += f'{f.cuda_src}\n' \
                    if compile_mode == 'cuda' else f.llvm_src

        translated_src += '\nextern "C" {\n'
        for f in self.kernels:
            f.build()
            build_func = f.use_cuda if compile_mode == 'cuda' else f.use_llvm
            if build_func:
                translated_src += f'{f.cuda_src}\n' \
                    if compile_mode == 'cuda' else f.llvm_src
        translated_src += '}\n'

        check_folder(args_path)
        with open(args_path, 'wb') as f:
            pkl.dump({
                func.name: func.args
                for func in self.funcs + self.kernels
            }, f)
        check_folder(out_src_path)
        with open(out_src_path, 'w') as f:
            f.write(translated_src)
        check_folder(build_path)
        if compile_mode == 'cuda':
            res_code = self.compile_cuda(translated_src, build_path)
        else:
            res_code = self.compile_llvm(translated_src, build_path)
        if res_code == 0:
            check_folder(hash_path)
            with open(hash_path, 'wb') as f:
                f.write(cur_hash)
            return True     # success
        return False        # failed

    def compile_cuda(self, translated_src, build_path):
        return jit_cu_lib.lib.cuda_compile_program(
            c_char_p(translated_src.encode()),
            c_int(86),
            c_char_p(zpc_include_path.encode()),
            c_bool(False),
            c_bool(True),
            c_bool(True),
            c_bool(True),
            c_char_p(build_path.encode()))

    def compile_llvm(self, translated_src, build_path):
        return jit_llvm_lib.lib.cpp_compile_program(
            c_char_p(translated_src.encode()),
            c_char_p(zpc_include_path.encode()),
            c_char_p(build_path.encode())
        )

    def register_func(self, func):
        self.funcs.append(func)

    def register_kernel(self, kernel):
        self.kernels.append(kernel)

    def register_sym_func(self, sym_func):
        self.sym_funcs.append(sym_func)

    def hash(self) -> int:
        h = hashlib.sha256()
        for f in self.funcs:
            f.update_hash(h)
        for f in self.kernels:
            f.update_hash(h)
        for f in self.sym_funcs:
            f.update_hash(h)
        return h.digest()


def get_module(module_name):
    module = registered_modules.get(module_name)
    if module:
        return module
    zs_module = Module(module_name)
    registered_modules[module_name] = zs_module
    return zs_module
