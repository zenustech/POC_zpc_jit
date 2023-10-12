import inspect
import sympy
import numpy as np
from .symbols import mat_size
from ..types import get_module
from ..context import registered_functions

DenseMat = sympy.matrices.dense.MutableDenseMatrix
NDenseMat = sympy.tensor.array.dense_ndim_array.MutableDenseNDimArray


def translate_sym_expr(expr, elem_type):
    def parse_bin_op(expr, op_char):
        return op_char.join(f'({translate_sym_expr(arg, elem_type)})'
                            for arg in expr.args)

    def parse_func_op(expr, func_name):
        args_str = ','.join(f'({translate_sym_expr(arg, elem_type)})'
                            for arg in expr.args)
        return f'zs::{func_name}({args_str})'

    def parse_Mul(expr):
        return parse_bin_op(expr, '*')

    def parse_Add(expr):
        return parse_bin_op(expr, '+')

    def parse_Pow(expr):
        return parse_func_op(expr, 'pow')

    def parse_Number(expr):
        return f'({elem_type.name})' + str(expr)

    def parse_Symbol(expr):
        expr_str = str(expr)
        if '@' in expr_str:
            words = expr_str.split('@')
            vec_name = words[0]
            vec_index = ','.join(words[1:])
            return f'{vec_name}({vec_index})'
        else:
            return expr_str

    type2func = {
        sympy.core.mul.Mul: parse_Mul,
        sympy.core.add.Add: parse_Add,
        sympy.core.power.Pow: parse_Pow,
        sympy.core.numbers.Number: parse_Number,
        sympy.core.symbol.Symbol: parse_Symbol
    }

    for expr_type, parse_func in type2func.items():
        if isinstance(expr, expr_type):
            return parse_func(expr)


def expr_to_vec(expr):
    if isinstance(expr, DenseMat):
        return expr.reshape(mat_size(expr), 1)
    return DenseMat([expr])


def vars_to_vec(vars):
    assert (isinstance(vars, list) or isinstance(vars, tuple))
    cat_vars = None
    for var in vars:
        vec_var = expr_to_vec(var)
        cat_vars = vec_var if cat_vars is None else cat_vars.col_join(vec_var)
    return cat_vars


def sym_diff(expr, vars):
    if isinstance(vars, tuple) or isinstance(vars, list):
        cat_vars = vars_to_vec(vars)
    else:
        cat_vars = expr_to_vec(vars)
    expr = expr_to_vec(expr)

    print(type(sympy.diff(expr, cat_vars)))
    return sympy.diff(expr, cat_vars)[:, 0, :, 0].transpose()


def zs_mat_typename(shape, elem_type, mode='svec'):
    assert (mode in ['svec', 'arr'])
    vec_name = 'vec' if mode == 'svec' else 'arr'
    return f'zs::{vec_name}<{elem_type.name}, {", ".join(str(x) for x in shape)}>'


def sym_name(symbol):
    if isinstance(symbol, DenseMat) or isinstance(symbol, NDenseMat):
        return str(symbol[0]).split('@')[0]
    return str(symbol)

# TODO -> get_sym_func_src + sym_func


def get_sym_func_src(func_name: str, expr, vars, elem_type):
    vars = vars \
        if isinstance(vars, tuple) or isinstance(vars, list) \
        else [vars]
    # used to create function header

    var_names = [sym_name(var) for var in vars]
    arg_decls = [f'T_{i}&& {str(arg)}' for i, arg in enumerate(var_names)]
    # T_i&& arg_i
    func_header = f'template<{", ".join(f"class T_{i}" for i in range(len(var_names)))}>\n' +\
        f'__device__ __host__  constexpr auto {func_name}({", ".join(arg_decls)})' + '{\n'
    func_tail = '\treturn __zs_gen_ret;\n}'
    func_body = ''
    if isinstance(expr, DenseMat) or isinstance(expr, NDenseMat):
        ret_shape = expr.shape
        ret_type_str = zs_mat_typename(ret_shape, elem_type)
        func_body = '\t' + ret_type_str + ' __zs_gen_ret{};\n'
        # __zs_gen_ret
        for idx, val_expr in np.ndenumerate(expr):
            func_body += \
                f'\t__zs_gen_ret{idx} = {translate_sym_expr(val_expr, elem_type)};\n'
    else:
        func_body = '\t' + elem_type.name + ' __zs_gen_ret{};\n'
        func_body += f'\t__zs_gen_ret = {translate_sym_expr(expr, elem_type)};\n'

    return func_header + func_body + func_tail

# TODO: add svec mode and arr mode


# TODO: handle llvm 
class SymFunc:
    def __init__(self, name, module, expr, vars, elem_type, 
                 use_cuda: bool=True, use_llvm: bool=False) -> None:
        self.name = name
        self.module = module
        self.expr = expr
        self.vars = vars
        self.elem_type = elem_type
        self.src = str(expr)
        self.cuda_src = None
        self.use_cuda = use_cuda
        self.use_llvm = use_llvm

    def update_hash(self, h):
        h.update(bytes(self.src, 'utf-8'))

    def build(self):
        self.cuda_src = get_sym_func_src(
            self.name, self.expr, self.vars, self.elem_type)


def gen_sym_func(name, module_name, expr, vars, elem_type):
    module = get_module(module_name)
    sym_func = SymFunc(name, module, expr, vars, elem_type)
    registered_functions[name] = sym_func
    module.register_sym_func(sym_func)
    return sym_func
