import inspect
import ast
from .context import registered_functions
from .containers import TileVectorViewType, DataType, ViewType
from .config import has_omp_lib

ast_to_op_char = {
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Eq: "==",
    ast.NotEq: "!=",

    ast.And: "&&",
    ast.Or: "||",

    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "/",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.BitAnd: "&",
    ast.MatMult: "*",   # TODO: may change the usage of '@' in the future

    ast.USub: "-",
    ast.Not: "!",
    ast.Mod: "%"
}


ast_to_func_name = {
    ast.Pow: 'pow'
}


type_strs = [
    'float',
    'double',
    'int',
    'char',
    'string'
]


class BlockStmt:
    def __init__(self):
        self.vars = []  # var names


class ArgsInfo:
    def __init__(self, names, annotations, inds, tv_unnamed_list,
                 tv_unnamed_tags) -> None:
        self.names = names
        self.annotations = annotations
        self.inds = inds
        self.tv_unnamed_list = tv_unnamed_list
        self.tv_unnamed_tags = tv_unnamed_tags


class FunctionTranslator:
    def __init__(self, func):
        self.func = func
        self.src_code, self.func_line_no = inspect.getsourcelines(func)
        self.ast_root = ast.parse(inspect.getsource(func))
        self.name = self.func.__name__
        argspec = inspect.getfullargspec(self.func)
        self.args = argspec.args
        self.arg_inds = {arg: ind for ind, arg in enumerate(self.args)}
        self.args_annotations = argspec.annotations
        self.defaults = argspec.defaults if argspec.defaults else []
        self.header_stmt = ""
        self.tail_stmt = ""

    def translate(self, compile_mode: str = 'cuda'):
        self.blocks = [BlockStmt()]
        for arg in self.args:
            self.blocks[-1].vars.append(arg)
        self.indent_str = ""
        self.var_ind = 0
        self.label_ind = 0
        self.while_start_labels = []
        self.while_end_labels = []
        self.while_else_labels = []
        self.loop_types = []
        self.current_block = None
        self.parsing_indices = False
        self.indices_has_tuple = False
        self.compile_mode = compile_mode
        self.body = []
        self.translate_node(self.ast_root.body[0])
        if compile_mode == 'cuda':
            self.cuda_body = self.body
            self.cuda_src = '\n'.join(self.body) + '\n'
        else:
            self.llvm_body = self.body
            self.llvm_src = '\n'.join(self.body) + '\n'

    def translate_node(self, node, stmt_level=False):
        type2func = {
            ast.FunctionDef: self.parse_FunctionDef,

            ast.Name: self.parse_Name,
            ast.Attribute: self.parse_Attribute,
            ast.Constant: self.parse_Constant,
            ast.Tuple: self.parse_Tuple,

            ast.If: self.parse_If,
            ast.While: self.parse_While,
            ast.For: self.parse_For,
            ast.Break: self.parse_Break,

            ast.Compare: self.parse_Compare,
            ast.BoolOp: self.parse_BoolOp,
            ast.BinOp: self.parse_BinOp,
            ast.UnaryOp: self.parse_UnaryOp,
            ast.Expr: self.parse_Expr,
            ast.Call: self.parse_Call,
            ast.Subscript: self.parse_Subscript,
            ast.Assign: self.parse_Assign,
            ast.Return: self.parse_Return,
            ast.AugAssign: self.parse_AugAssign,

            ast.Pass: self.parse_Pass
        }

        params = [node]
        if type(node) == ast.Call:
            params.append(not stmt_level)
        if type(node) not in type2func:
            raise RuntimeError(f"ast node of type {type(node)} not supported")
        return type2func[type(node)](*params)

    def indent(self):
        self.indent_str += "\t"

    def dedent(self):
        self.indent_str = self.indent_str[:-1]

    def begin_block(self):
        self.add_line("{")
        self.blocks.append(BlockStmt())
        self.indent()

    def end_block(self):
        self.dedent()
        self.blocks.pop()
        self.add_line("}")

    def parse_Pass(self, node: ast.Pass):
        # NOTE: ignore pass for now
        pass

    def parse_If(self, node: ast.If):
        test_var = self.translate_node(node.test)
        self.add_line(f"if ((bool){test_var})")
        self.begin_block()
        for n in node.body:
            self.translate_node(n, stmt_level=True)
        self.end_block()
        if len(node.orelse):
            self.add_line("else")
            self.begin_block()
            for n in node.orelse:
                self.translate_node(n, stmt_level=True)
            self.end_block()

    def parse_While(self, node: ast.While):
        start_label, end_label, else_label = self.push_while_label()
        self.add_line(f"{start_label}:;")
        test_var = self.translate_node(node.test)
        self.add_line(f"if (!({test_var})) goto {else_label};")
        self.loop_types.append("while")
        self.begin_block()
        for n in node.body:
            self.translate_node(n, stmt_level=True)
        self.end_block()
        self.loop_types.pop()
        self.add_line(f"goto {start_label};")
        self.add_line(f"{else_label}:;")
        self.begin_block()
        for n in node.orelse:
            self.translate_node(n, stmt_level=True)
        self.end_block()
        self.add_line(f"{end_label}:;")

    def parse_Break(self, node: ast.Break):
        if self.loop_types:
            if self.loop_types[-1] == "while":
                self.add_line(f"goto {self.while_end_labels[-1]};")
            else:
                self.add_line("break;")
        else:
            raise RuntimeError("cannot break outside of loops")

    def parse_For(self, node: ast.For):
        # ignore orelse
        target_node = node.target
        iter_node = node.iter
        iterator_var = self.translate_node(iter_node)
        target_name: str
        vars = []
        if type(target_node) == ast.Name:
            target_name = target_node.id
            vars = [target_name]
        elif type(target_node) == ast.Tuple:
            for arg in target_node.elts:
                assert (type(arg) == ast.Name)
                vars.append(arg.id)
            target_name = "[" + ",".join(vars) + "]"

        self.add_line(f"for (auto&& {target_name} : {iterator_var}) ")
        self.begin_block()
        for var in vars:
            self.add_var(var)
        self.loop_types.append("for")
        for n in node.body:
            self.translate_node(n, stmt_level=True)
        self.end_block()
        self.loop_types.pop()

    def add_var(self, name=None):
        vars = self.blocks[-1].vars
        if name:
            if name in vars:
                raise RuntimeError("multiple definitions of variables!")
            vars.append(name)
            return name
        var = f"tmp_var_{self.var_ind}"
        vars.append(var)
        self.var_ind += 1
        return var

    def push_while_label(self):
        start_label, end_label, else_label = f"while_start_{self.label_ind}", \
            f"while_end_{self.label_ind}", f"while_else_{self.label_ind}"
        self.while_start_labels.append(start_label)
        self.while_end_labels.append(end_label)
        self.while_else_labels.append(else_label)
        self.label_ind += 1
        return start_label, end_label, else_label

    def pop_while_label(self):
        self.while_start_labels.pop()
        self.while_end_labels.pop()
        self.while_else_labels.pop()

    def has_var(self, name):
        if name == 'tid' and self.compile_mode == 'llvm':
            return True
        for block in self.blocks[::-1]:
            if name in block.vars:
                return True
        return False

    def parse_Compare(self, node: ast.Compare):
        left_var = self.translate_node(node.left)
        result_var = None
        for op, cmpvar in zip(node.ops, node.comparators):
            cmpvar = self.translate_node(cmpvar)
            cmp_res = self.add_op_call(type(op), [left_var, cmpvar])
            if result_var is None:
                result_var = cmp_res
            else:
                result_var = self.add_op_call(ast.And, [result_var, cmp_res])
        return result_var

    def parse_BoolOp(self, node: ast.BoolOp):
        vars = [self.translate_node(n) for n in node.values]
        return self.add_op_call(type(node.op), vars)

    def parse_BinOp(self, node: ast.BinOp):
        left_var = self.translate_node(node.left)
        right_var = self.translate_node(node.right)
        return self.add_op_call(type(node.op), [left_var, right_var])

    def parse_UnaryOp(self, node: ast.UnaryOp):
        operand = self.translate_node(node.operand)
        return self.add_op_call(type(node.op), [operand])

    def parse_Expr(self, node: ast.Expr):
        return self.translate_node(node.value, True)

    def parse_Call(self, node: ast.Call, use_ret_val=True):
        is_method = False
        func_node = node.func
        if isinstance(func_node, ast.Name):
            func_val = self.parse_Name(func_node, False)
            is_type = isinstance(func_val, DataType) and func_val.in_kernel
            func_name = func_val.name if is_type else func_val 
            is_method = is_type
        if isinstance(node.func, ast.Attribute):
            func_name = self.translate_node(node.func)
            is_method = True 
        return self.add_call(
            func_name, [
                self.translate_Call_Args(arg) for arg in node.args],
            use_ret_val, is_method=is_method)

    def translate_Call_Args(self, node):
        if isinstance(node, ast.Tuple) or isinstance(node, ast.List):
            is_value_seq = True
            for elt in node.elts:
                if not (
                        isinstance(elt, ast.Constant) and
                        isinstance(elt.value, int)):
                    is_value_seq = False
                    break
            if is_value_seq:
                return f'value_seq<{",".join(self.parse_Constant(elt) for elt in node.elts)}>' + '{}'

            is_type_seq = True
            for elt in node.elts:
                if not isinstance(elt, ast.Name):
                    is_type_seq = False
                    break
                if elt.id not in type_strs:
                    is_type_seq = False
                    break
            if is_type_seq:
                return f'zs::type_seq<{",".join(self.parse_Name(elt) for elt in node.elts)}>' + '{}'

            def wrap_if_is_type(name):
                if name in type_strs:
                    return f'zs::wrapt<{name}>' + '{}'
                return name

            # make_tuple
            return f'zs::make_tuple({",".join(wrap_if_is_type(self.translate_node(elt)) for elt in node.elts)})'

        elif isinstance(node, ast.Name):
            if node.id in type_strs:
                # handle float, double, char, int
                return f'wrapt<{node.id}>' + '{}'
            else:
                return self.parse_Name(node)
        else:
            return self.translate_node(node)

    def has_func(self, func_name):
        func = registered_functions.get(func_name)
        if func is None:
            return False
        if self.compile_mode == 'cuda':
            return func.use_cuda
        else:
            return func.use_llvm

    # def get_func(self, node):
    #     if type(node) == ast.Name:
    #         func_name = node.id
    #         if func_name in registered_functions:
    #             return registered_functions[func_name]
    #         raise RuntimeError(f"cannot find function {func_name}")
    #     else:
    #         raise NotImplementedError()

    def parse_Subscript(self, node: ast.Subscript, lval=False):
        val = self.translate_node(node.value)
        self.parsing_indices = True
        self.indices_has_tuple = False
        if isinstance(node.slice, ast.Name):
            var = self.add_var()
            self.add_line(f"auto&& {var} = {val}[{node.slice.id}]; ")
            return var
        if isinstance(node.slice, ast.Constant):
            slice_var = self.parse_Constant(node.slice)
            var = self.add_var()
            self.add_line(f'auto&& {var} = {val}[{slice_var}];')
            return var
        indices = [self.translate_node(arg) for arg in node.slice.elts]

        if isinstance(node.value, ast.Name):
            container_name = node.value.id
            if container_name in self.args_annotations:
                if isinstance(
                        self.args_annotations[container_name],
                        TileVectorViewType):
                    tag_elt = node.slice.elts[0]
                    if isinstance(tag_elt, ast.Constant) and isinstance(tag_elt.value, str):
                        prop_tag = tag_elt.value
                        if container_name not in self.tv_unnamed_list:
                            self.tv_unnamed_list.append(container_name)
                            self.tv_unnamed_tags[container_name] = []
                        if prop_tag not in self.tv_unnamed_tags[container_name]:
                            self.tv_unnamed_tags[container_name].append(
                                prop_tag)
                        if len(node.slice.elts) == 3:
                            dim_elt = node.slice.elts[1]
                            assert(isinstance(dim_elt, ast.Constant))
                            prop_dim = dim_elt.value
                            assert(isinstance(prop_dim, int))
                            indices = [f'__zs_gen_tag_offset_{container_name}_{prop_tag} + {prop_dim}',
                                        indices[2]]
                        else:
                            indices[0] = f'__zs_gen_tag_offset_{container_name}_{prop_tag}'

        if self.indices_has_tuple:
            method_name = ".tuple" if lval else ".pack"
        else:
            method_name = ""
        self.parsing_indices = False
        self.indices_has_tuple = False
        if type(node.slice) == ast.Tuple:
            expr_str = f"{val}{method_name}(" + ",".join(indices) + ")"
            self.indices_has_tuple = False
        else:
            expr_str = f"{val}[" + ",".join(indices) + "]"
        if lval:
            return expr_str
        var = self.add_var()
        self.add_line(f"auto&& {var} = {expr_str}; ")
        return var

    def parse_Tuple(self, node: ast.Tuple):
        tuple_str = ",".join([self.translate_node(n) for n in node.elts])
        if self.parsing_indices:
            self.indices_has_tuple = True
            return f"zs::dim_c<{tuple_str}>"
        else:
            raise NotImplementedError()

    def check_var(self, var):
        if not self.has_var(var):
            self.raise_undefined_var(var)

    def raise_undefined_var(self, var):
        raise RuntimeError(f"undefined reference {var}")

    def parse_Assign(self, node: ast.Assign):
        assert (len(node.targets) == 1)
        target_node = node.targets[0]
        if type(target_node) == ast.Tuple:
            raise NotImplementedError()
        elif type(target_node) == ast.Subscript:
            # will not check the left var
            lfs = self.parse_Subscript(target_node, True)
            rhs = self.translate_node(node.value)
            self.add_line(f"{lfs} = {rhs};")
        elif type(target_node) == ast.Name:
            lfs = target_node.id
            rhs = None
            if type(node.value) == ast.Name:
                rhs = node.value.id
                self.check_var(rhs)
            else:
                rhs = self.translate_node(node.value)
            if self.has_var(lfs):
                self.add_line(f"{lfs} = {rhs}; ")
            else:
                self.add_var(lfs)
                self.add_line(f"auto {lfs} = {rhs}; ")
        elif type(target_node) == ast.Attribute:
            raise NotImplementedError()

    def parse_Return(self, node: ast.Return):
        var = self.translate_node(node.value)
        self.add_line(f"return {var}; ")

    def parse_AugAssign(self, node: ast.AugAssign):
        target_str = self.translate_node(node.target)
        target_val = self.translate_node(node.target)
        var_val = self.translate_node(node.value)
        var_res = self.add_op_call(type(node.op), [
            target_val, var_val])
        self.add_line(f"{target_str} = {var_res}; ")

    def add_line(self, line):
        self.body.append(f'{self.indent_str}{line}')

    def add_call(self, func_name, vars, use_ret_val=True, is_method=False):
        if (not self.has_func(func_name)) and func_name:
            func_name = func_name if is_method else f'zs::{func_name}'
        if use_ret_val:
            var = self.add_var()
            self.add_line(
                f"auto&& {var} = {func_name}({','.join(vars)}); ")
            return var
        else:
            self.add_line(f"{func_name}({','.join(vars)}); ")
            return None

    def add_op_call(self, op_ast, vars):
        if op_ast in ast_to_op_char:
            op_char = ast_to_op_char[op_ast]
            if len(vars) == 1:
                var = self.add_var()
                self.add_line(f'auto&& {var} = {op_char}{vars[0]}; ')
                return var
            elif len(vars) == 2:
                var = self.add_var()
                self.add_line(f'auto&& {var} = {vars[0]}{op_char}{vars[1]}; ')
                return var
            else:
                raise RuntimeError(
                    f'too many parameters for operator {op_char}: {vars}')
        else:
            return self.add_call(ast_to_func_name[op_ast], vars)

    def parse_FunctionDef(self, node: ast.FunctionDef):
        self.tv_unnamed_list = []
        self.tv_unnamed_tags = {}
        self.add_line("{")
        self.indent_str = "\t"
        for n in node.body:
            self.translate_node(n, stmt_level=True)
        self.indent_str = ""
        self.add_line("}")

        if 'return' not in self.args_annotations:
            ret_decl = 'auto'
        else:
            ret_ann = self.args_annotations['return']
            ret_decl = 'void' if ret_ann is None else ret_ann.name
        template_header = "template<" + \
            ",".join([f"class T_{i}" for i in range(len(self.args))]) + ">"
        offset_var_names = []
        for tv in self.tv_unnamed_list:
            offset_var_names += [
                f'__zs_gen_tag_offset_{tv}_{attr}'
                for attr in self.tv_unnamed_tags[tv]]
        if self.compile_mode == 'cuda':
            func_decl = f"__device__ __host__ {ret_decl} {self.name} (" + ",".join(
                [f"T_{i}&& {arg}" for i, arg in enumerate(self.args)] +
                [f'zs::size_t {offset_var_name}'
                 for offset_var_name in offset_var_names]) + ")"
            self.cuda_header = template_header + '\n' + func_decl + '\n'
        else:
            func_decl = f"{ret_decl} {self.name} (" + ",".join(
                [f"T_{i}&& {arg}" for i, arg in enumerate(self.args)] +
                [f'zs::size_t {offset_var_name}'
                 for offset_var_name in offset_var_names]) + ")"
            self.llvm_header = template_header + '\n' + func_decl + '\n'

    def parse_Name(self, node: ast.Name, to_str=True):
        if not self.has_var(node.id):
            # if is a zpc type in globals or __closure__/co_freevars
            # TODO: handle __closure__/co_freevars
            global_var = self.func.__globals__.get(node.id)
            if global_var is not None:
                if isinstance(global_var, DataType):
                    return global_var.name if to_str else global_var
            # TODO: check builtin type names
            # TODO: check zpc builtin names...
        return node.id

    def parse_Attribute(self, node: ast.Attribute):
        if isinstance(node.value, ast.Name):
            val_name = self.parse_Name(node.value, False)
            if isinstance(val_name, DataType) and val_name.in_kernel:
                return f'{val_name.name}::{node.attr}'
            return f"{node.value.id}.{node.attr}"
        return self.translate_node(node.value) + f".{node.attr}"

    def parse_Constant(self, node: ast.Constant):
        if isinstance(node.value, str):
            return self.handle_quote(ast.unparse(node))
        else:
            return f'{node.value}'

    def handle_quote(self, string):
        if string[0] == "'" and string[-1] == "'":
            return '"' + string[1:-1] + '"'

    @property
    def args_info(self) -> ArgsInfo:
        return ArgsInfo(self.args, self.args_annotations, self.arg_inds,
                        self.tv_unnamed_list, self.tv_unnamed_tags)


class KernelTranslator(FunctionTranslator):
    def __init__(self, func, llvm_launch_symbol_name: str):
        super().__init__(func)
        for arg in self.args:
            if arg not in self.args_annotations:
                raise RuntimeError(
                    f'The type of argument {arg} should be annotated!')
        self.llvm_launch_symbol_name = llvm_launch_symbol_name

    def translate(self, compile_mode: str = 'cuda'):
        super().translate(compile_mode)
        if compile_mode == 'llvm':
            self.llvm_src += self.llvm_launch_src

    def parse_FunctionDef(self, node: ast.FunctionDef):
        self.tv_unnamed_list = []
        self.tv_unnamed_tags = {}
        self.add_line("{")
        self.indent_str = "\t"
        for n in node.body:
            self.translate_node(n, stmt_level=True)
        self.indent_str = ""
        self.add_line("}")

        arg_decls = []
        view_ptr_arg_decls = []
        for ind, arg in enumerate(self.args):
            arg_type = self.args_annotations[arg]
            decl = f'{arg_type.name} {arg}'
            view_ptr_arg_decl = f'{arg_type.name}& {arg}' \
                if isinstance(arg_type, ViewType) else decl
            default_ind = len(self.args) - ind - 1
            if default_ind < len(self.defaults):
                decl += f' = {self.defaults[default_ind]}'
                view_ptr_arg_decl += f' = {self.defaults[default_ind]}'
            arg_decls.append(decl)
            view_ptr_arg_decls.append(view_ptr_arg_decl)
        offset_var_names = []
        for tv in self.tv_unnamed_list:
            offset_var_names += [
                f'__zs_gen_tag_offset_{tv}_{attr}'
                for attr in self.tv_unnamed_tags[tv]]
        offset_var_decls = [
            f'zs::size_t {offset_var_name}'
            for offset_var_name in offset_var_names]

        if self.compile_mode == 'cuda':
            func_decl = f"__global__ void {self.name} ({','.join(arg_decls + offset_var_decls)})"
            self.cuda_header = func_decl + '\n'
        else:
            func_decl = f"void {self.name} ({','.join(view_ptr_arg_decls + offset_var_decls + ['zs::size_t tid = 0'])})"
            self.llvm_header = func_decl + '\n'
            omp_preproc_str = '\n#pragma omp parallel for' if has_omp_lib else ''
            self.llvm_launch_src = f"void {self.llvm_launch_symbol_name} ({','.join(view_ptr_arg_decls + offset_var_decls + ['zs::size_t __zs_gen_num_threads = 0'])})" + \
                '{' + omp_preproc_str + '\n\tfor (zs::size_t tid = 0; tid < __zs_gen_num_threads; tid++)' + \
                self.name + '(' + ','.join(self.args +
                                           offset_var_names + ['tid']) + ');\n}\n'
