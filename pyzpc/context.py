registered_functions = {}
registered_kernels = {}
registered_modules = {}


def clear_context():
    global registered_functions, registered_kernels, registered_modules
    registered_functions.clear()
    registered_kernels.clear()
    registered_modules.clear()
