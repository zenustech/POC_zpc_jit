import ctypes
from ctypes import c_char_p, c_void_p, c_int, c_ubyte, c_size_t, c_int8, POINTER
from .utils import CLib, str2ctype

zpc_lib = CLib()

class MemSrc:
    um = None
    host = None
    device = None
    ind2str = ['host', 'device', 'um']

    @staticmethod
    def from_name(name):
        if name not in ('um', 'device', 'host'):
            raise RuntimeError(
                f'memsrc should be one of um, device, host, got: {name}')
        # return getattr(MemSrc, name)
        return getattr(MemSrc, name)

    @staticmethod
    def to_name(ind):
        return MemSrc.ind2str[ind]


def init_zpc_lib(path: str):
    zpc_lib.init_lib(path)
    setattr(MemSrc, 'um', zpc_lib.lib.mem_enum__um())
    setattr(MemSrc, 'host', zpc_lib.lib.mem_enum__host())
    setattr(MemSrc, 'device', zpc_lib.lib.mem_enum__device())
    # MemSrc.um = zpc_lib.lib.mem_enum__um()
    # MemSrc.host = zpc_lib.lib.mem_enum__host()
    # MemSrc.device = zpc_lib.lib.mem_enum__device()
    for elem_type in ('int', 'float', 'double'):
        for vir_suf in ['', '_virtual']:
            for length in (8, 32, 64, 512):
                zpc_lib.register(c_void_p, f'pyview__tv_{elem_type}_{length}{vir_suf}', c_void_p)
                zpc_lib.register(None, f'del_pyview__tv_{elem_type}_{length}', c_void_p)
                zpc_lib.register(c_void_p, f'pyview__tvn_{elem_type}_{length}{vir_suf}', c_void_p)
                zpc_lib.register(None, f'del_pyview__tvn_{elem_type}_{length}', c_void_p)

                zpc_lib.register(c_void_p, f'container__tv_{elem_type}_{length}{vir_suf}', c_void_p, c_void_p, c_size_t)
                zpc_lib.register(None, f'del_container__tv_{elem_type}_{length}{vir_suf}', c_void_p)
                zpc_lib.register(None, f'relocate_container__tv_{elem_type}_{length}{vir_suf}', c_void_p, c_int, c_int8)
                zpc_lib.register(c_int, f'property_offset__tv_{elem_type}_{length}{vir_suf}', c_void_p, c_char_p)
                zpc_lib.register(None, f'resize_container__tv_{elem_type}_{length}{vir_suf}', c_void_p, c_size_t)
        
            zpc_lib.register(c_void_p, f'pyview__v_{elem_type}{vir_suf}', c_void_p)
            zpc_lib.register(None, f'del_pyview__v_{elem_type}', c_void_p)
            zpc_lib.register(c_void_p, f'container__v_{elem_type}{vir_suf}', c_void_p, c_size_t)
            zpc_lib.register(None, f'del_container__v_{elem_type}{vir_suf}', c_void_p)
        
            zpc_lib.register(None, f'relocate_container__v_{elem_type}{vir_suf}', c_void_p, c_int, c_int8)
            zpc_lib.register(None, f'resize_container__v_{elem_type}{vir_suf}', c_void_p, c_size_t)

        zpc_lib.register(str2ctype[elem_type], f'get_val_container__v_{elem_type}', c_void_p)
        zpc_lib.register(None, f'set_val_container__v_{elem_type}', c_void_p, str2ctype[elem_type])


    for mem_src in ('host', 'device', 'um'):
        zpc_lib.register(c_ubyte, f'mem_enum__{mem_src}')

    zpc_lib.register(c_void_p, 'allocator', c_int, c_int8)
    zpc_lib.register(None, 'del_allocator', c_void_p)

    zpc_lib.register(c_void_p, 'property_tags', POINTER(c_char_p), POINTER(c_int), c_size_t)
    zpc_lib.register(c_void_p, 'del_property_tags', c_void_p)
    zpc_lib.register(None, 'property_tags_get_item', c_void_p, c_size_t, POINTER(c_char_p), POINTER(c_size_t))
    zpc_lib.register(c_size_t, 'property_tags_get_size', c_void_p)

    zpc_lib.register(c_void_p, 'policy__device')
    zpc_lib.register(None, 'del_policy__device', c_void_p)
    zpc_lib.register(None, 'launch__device', c_void_p, c_void_p, c_size_t, POINTER(c_void_p))

    def def_small_vec_funcs(api_suffix):
        zpc_lib.register(c_void_p, f'small_vec__{api_suffix}')
        zpc_lib.register(None, f'del_small_vec__{api_suffix}', c_void_p)
        zpc_lib.register(c_void_p, f'small_vec_data_ptr__{api_suffix}', c_void_p)

    for elem_type in ('int', 'float', 'double'):
        def_small_vec_funcs(elem_type)
        for m in range(1, 5):
            def_small_vec_funcs(f'{elem_type}_{m}')
            for n in range(1, 5):
                def_small_vec_funcs(f'{elem_type}_{m}_{n}')
    