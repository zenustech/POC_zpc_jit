from .types import Function, Kernel, get_module
from .context import registered_functions, registered_kernels
from functools import partial


def func(f, use_cuda: bool = True, use_llvm: bool = True):
    name = f.__name__
    # TODO: to support ret type annotation
    module = get_module(f.__module__)
    func = Function(name, module,
                    ret=False,
                    func=f,
                    use_cuda=use_cuda,
                    use_llvm=use_llvm)
    registered_functions[name] = func
    module.register_func(func)
    return func


def kernel(f, use_cuda: bool = True, use_llvm: bool = False):
    name = f.__name__
    module = get_module(f.__module__)
    kernel = Kernel(name, module,
                    func=f,
                    use_cuda=use_cuda,
                    use_llvm=use_llvm)
    registered_kernels[name] = kernel
    module.register_kernel(kernel)
    return kernel


cuda_func = partial(func, use_cuda=True, use_llvm=False)
llvm_func = partial(func, use_cuda=False, use_llvm=True)
cuda_kernel = partial(kernel, use_cuda=True, use_llvm=False)
llvm_kernel = partial(kernel, use_cuda=False, use_llvm=True)
