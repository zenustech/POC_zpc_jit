from .utils import CLib
from .context import clear_context
from ctypes import c_char_p, c_uint32, c_uint64, c_int, c_float, c_double, \
    c_bool, c_size_t, POINTER, c_void_p, c_int8

zeno_lib = CLib() 

def init_zeno_lib(lib_path):
    clear_context()
    global zeno_lib
    zeno_lib.init_lib(lib_path)

    # Zeno_DestroyObject 
    zeno_lib.register(c_uint32, 'Zeno_DestroyObject', c_uint64)
    # APIs for creating svec
    zeno_lib.register(c_uint32, 'ZS_GetObjectZsVecData', c_uint64, POINTER(c_void_p), POINTER(c_size_t), POINTER(c_size_t), 
                      POINTER(c_size_t), POINTER(c_int))
    for type_str in ['float', 'double', 'int']:
        zeno_lib.register(c_uint32, f'ZS_CreateObjectZsSmallVec_{type_str}_scalar', POINTER(c_uint64))
        for dim_i in range(1, 5):
            zeno_lib.register(c_uint32, f'ZS_CreateObjectZsSmallVec_{type_str}_{dim_i}', POINTER(c_uint64))
            for dim_j in range(1, 5):
                zeno_lib.register(c_uint32, f'ZS_CreateObjectZsSmallVec_{type_str}_{dim_i}x{dim_j}', POINTER(c_uint64))
    
        # APIs for creating vector 
        zeno_lib.register(c_uint32, f'container_obj__v_{type_str}', POINTER(c_uint64), c_void_p, c_size_t)
        zeno_lib.register(c_uint32, f'container_obj__v_{type_str}_virtual', POINTER(c_uint64), c_void_p, c_size_t)
        for length in [8, 32, 64, 512]:
            zeno_lib.register(c_uint32, f'container_obj__tv_{type_str}_{length}', POINTER(c_uint64), c_void_p, c_void_p, c_size_t)
            zeno_lib.register(c_uint32, f'container_obj__tv_{type_str}_{length}_virtual', POINTER(c_uint64), c_void_p, c_void_p, c_size_t)
    zeno_lib.register(c_uint32, 'ZS_GetTileVectorData', c_uint64, POINTER(c_void_p), POINTER(c_int), c_void_p, 
                      POINTER(c_size_t), POINTER(c_int), POINTER(c_size_t), POINTER(c_int8), POINTER(c_bool))
    zeno_lib.register(c_uint32, 'ZS_GetVectorData', c_uint64, POINTER(c_void_p), POINTER(c_int), 
                      POINTER(c_size_t), POINTER(c_int), POINTER(c_int8), POINTER(c_bool))
    