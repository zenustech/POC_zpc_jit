from .zpc import zpc_lib
import numpy as np
from functools import reduce
import ctypes
from .utils import ctype2str
from ctypes import c_int, c_int8, c_char_p, c_size_t, c_double, c_float, \
    c_int32, c_int64, byref, c_uint64
from .zeno import zeno_lib


class DataType:
    def __init__(self, name, c_type=None, in_kernel=False) -> None:
        self.name = name
        self.c_type = c_type
        self.in_kernel = in_kernel


class ContainerType(DataType):
    pass


class ViewType(DataType):
    pass


class Container:
    pass


f64 = DataType('zs::f64', c_double)
f32 = DataType('zs::f32', c_float)
i32 = DataType('zs::i32', c_int32)
i64 = DataType('zs::i64', c_int64)

integer = DataType('int', c_int)
fl = DataType('float', c_float)
dbl = double = DataType('double', c_double)
string = DataType('zs::SmallString', c_char_p)

ind2zstype = [integer, fl, dbl]
zstype2nptype = {
    double: np.dtype(np.float64),
    fl: np.dtype(np.float32),
    integer: np.dtype(np.int32),
}
nptype2zstype = {v: k for k, v in zstype2nptype.items()}


class ZenoObject:
    def __init__(self, handle) -> None:
        zeno_lib.call('Zeno_ObjectIncReference', c_uint64(handle))
        self.handle = handle

    @staticmethod
    def from_handle(handle):
        return ZenoObject(handle)

    def as_type(self, obj_type):
        # vector variant...
        # svec variant...
        return obj_type.from_handle(self.handle)

    def to_handle(self):
        return self.handle

    def __del__(self):
        zeno_lib.call('Zeno_DestroyObject', c_uint64(self.handle))


class MemSrc:
    um = zpc_lib.lib.mem_enum__um()
    host = zpc_lib.lib.mem_enum__host()
    device = zpc_lib.lib.mem_enum__device()
    ind2str = ['host', 'device', 'um']

    @staticmethod
    def from_name(name):
        if name not in ('um', 'device', 'host'):
            raise RuntimeError(
                f'memsrc should be one of um, device, host, got: {name}')
        return getattr(MemSrc, name)

    @staticmethod
    def to_name(ind):
        return MemSrc.ind2str[ind]


class PropTags:
    def __init__(self, prop_dict: dict = {}) -> None:
        names = []
        sizes = []
        for k, v in prop_dict.items():
            names.append(k.encode())
            sizes.append(v)
        n = len(names)
        self.ptr = zpc_lib.lib.property_tags(
            (c_char_p * n)(*names), (c_int * n)(*sizes), c_size_t(n))

    def to_dict(self):
        ret = {}
        for i in range(self.size):
            k, v = self[i]
            ret[k] = v
        return ret

    @property
    def size(self):
        return zpc_lib.call('property_tags_get_size', self.ptr)

    def __getitem__(self, ind: int):
        name = c_char_p()
        size = c_size_t()
        zpc_lib.call('property_tags_get_item', self.ptr,
                     c_size_t(ind), byref(name), byref(size))
        return name.value.decode(), size.value

    def __del__(self):
        zpc_lib.lib.del_property_tags(self.ptr)


class TileVecNamedView:
    def __init__(self, ptr, elem_type=fl, length=32, virtual=False) -> None:
        self.ptr = ptr
        self.elem_type = elem_type
        self.length = length
        self.virtual = virtual
        self.api_suffix = f'{elem_type.name}_{length}' + \
            ('_virtual' if virtual else '')

    def call_zpc(self, func_name, *args):
        return zpc_lib.call(f'{func_name}_{self.api_suffix}', *args)

    def __del__(self):
        self.call_zpc('del_pyview__tvn', self.ptr)


class TileVecView:
    def __init__(self, ptr, elem_type=fl, length=32, virtual=False) -> None:
        self.ptr = ptr
        self.elem_type = elem_type
        self.length = length
        self.virtual = virtual
        self.api_suffix = f'{elem_type.name}_{length}' + \
            ('_virtual' if virtual else '')

    def call_zpc(self, func_name, *args):
        return zpc_lib.call(f'{func_name}_{self.api_suffix}', *args)

    def __del__(self):
        self.call_zpc('del_pyview__tv', self.ptr)


class VecView:
    def __init__(self, ptr, elem_type=fl, virtual=False) -> None:
        self.ptr = ptr
        self.elem_type = elem_type
        self.virtual = virtual
        self.api_suffix = f'{elem_type.name}' + ('_virtual' if virtual else '')

    def call_zpc(self, func_name, *args):
        return zpc_lib.call(f'{func_name}_{self.api_suffix}', *args)

    def __del__(self):
        self.call_zpc('del_pyview__v', self.ptr)


class Allocator:
    def __init__(self) -> None:
        pass


class TileVectorAPI(Container):
    def __init__(
            self, ptr, memsrc, tags, size, elem_type=fl, length=32,
            dev_id=0, virtual=False) -> None:
        self.memsrc = memsrc
        self.dev_id = dev_id
        self.tags = tags
        self.elem_type = elem_type
        self.length = length
        self.virtual = virtual
        self.api_suffix = f'{elem_type.name}_{length}' + \
            ('_virtual' if virtual else '')
        self.ptr = ptr
        self._size = size

    @property
    def size(self):
        return self._size

    def resize(self, size: int):
        self._size = size
        self.call_zpc('resize_container__tv', self.ptr, c_size_t(size))

    def call_zpc(self, func_name, *args):
        return zpc_lib.call(f'{func_name}_{self.api_suffix}', *args)

    def view(self, named=False) -> TileVecView:
        if not named:
            return TileVecView(self.call_zpc('pyview__tv', self.ptr),
                               self.elem_type, self.length, self.virtual)
        else:
            return self.named_view()

    def named_view(self) -> TileVecNamedView:
        return TileVecNamedView(self.call_zpc('pyview__tvn', self.ptr),
                                self.elem_type, self.length, self.virtual)

    def relocate(self, memsrc: str, proc_id):
        self.memsrc = memsrc
        memsrc = MemSrc.from_name(memsrc)
        self.call_zpc('relocate_container__tv',
                      self.ptr, memsrc, c_int8(proc_id))

    def tag_offset(self, tag_name: str):
        return self.call_zpc('property_offset__tv',
                             self.ptr, c_char_p(tag_name.encode()))


class TileVector(TileVectorAPI):
    def __init__(self, memsrc, tags, size, elem_type=fl, length=32,
                 dev_id=0, virtual=False) -> None:
        c_memsrc = MemSrc.from_name(memsrc)
        allocator = zpc_lib.lib.allocator(c_memsrc, dev_id)
        if isinstance(tags, dict):
            c_tags = PropTags(tags)
        api_suffix = f'{elem_type.name}_{length}' + \
            ('_virtual' if virtual else '')
        ptr = zpc_lib.call(
            f'container__tv_{api_suffix}', allocator, c_tags.ptr,
            c_size_t(size))
        super().__init__(ptr, memsrc, tags, size, elem_type, length, dev_id, virtual)

    def __del__(self):
        self.call_zpc('del_container__tv', self.ptr)


def create_tilevector_object(memsrc, tags, size, elem_type, length, dev_id, virtual):
    _handle = c_uint64()
    c_memsrc = MemSrc.from_name(memsrc)
    prop_tags = PropTags(tags)
    allocator = zpc_lib.call('allocator', c_memsrc, dev_id)
    api_suffix = f'{elem_type.name}_{length}' + ('_virtual' if virtual else '')
    zeno_lib.call(f'container_obj__tv_{api_suffix}', byref(
        _handle), allocator, prop_tags.ptr, c_size_t(size))
    return _handle.value


def tilevector_info_from_handle(handle):
    prop_tags = PropTags()
    _ptr = ctypes.c_void_p()
    _memsrc = ctypes.c_int()
    _size = ctypes.c_size_t()
    _dtype = ctypes.c_int()
    _dev_id = ctypes.c_int8()
    _is_virtual = ctypes.c_bool()
    _length = ctypes.c_size_t()
    zeno_lib.call('ZS_GetTileVectorData', handle, byref(_ptr), byref(_memsrc),
                  prop_tags.ptr, byref(_size), byref(_dtype), byref(_length),
                  byref(_dev_id), byref(_is_virtual))
    return _ptr.value, MemSrc.to_name(_memsrc.value), prop_tags, _size.value, \
        ind2zstype[_dtype.value], _length.value, _dev_id.value, _is_virtual.value


class TileVectorObject(TileVectorAPI, ZenoObject):
    def __init__(self, memsrc, tags, size, elem_type=fl, length=32, dev_id=0,
                 virtual=False, handle=None, ptr=None) -> None:
        if handle is None:
            handle = create_tilevector_object(
                memsrc, tags, size, elem_type, length, dev_id, virtual)
        if ptr is None:
            ptr = tilevector_info_from_handle(handle)[0]
        ptr, memsrc, prop_tags, size, elem_type, length, dev_id, virtual = \
            tilevector_info_from_handle(handle)
        TileVectorAPI.__init__(self, ptr, memsrc, tags,
                               size, elem_type, length, dev_id, virtual)
        ZenoObject.__init__(self, handle)

    @staticmethod
    def from_handle(handle):
        ptr, memsrc, prop_tags, size, elem_type, length, dev_id, virtual = \
            tilevector_info_from_handle(handle)
        return TileVectorObject(memsrc, prop_tags.to_dict(), size, elem_type, length,
                                dev_id, virtual, handle, ptr)

    def __del__(self):
        ZenoObject.__del__(self)


class VectorAPI(Container):
    def __init__(
            self, ptr, memsrc, size, elem_type=fl, dev_id=0, virtual=False) -> None:
        self.memsrc = memsrc
        self.dev_id = dev_id
        self.elem_type = elem_type
        self.virtual = virtual
        self.api_suffix = f'{elem_type.name}' + ('_virtual' if virtual else '')
        self.ptr = ptr
        self._size = size

    @property
    def size(self):
        return self._size

    def call_zpc(self, func_name, *args):
        return zpc_lib.call(f'{func_name}_{self.api_suffix}', *args)

    def view(self):
        return VecView(self.call_zpc('pyview__v', self.ptr), self.elem_type,
                       self.virtual)

    def relocate(self, memsrc: str, proc_id):
        memsrc = MemSrc.from_name(memsrc)
        self.memsrc = memsrc
        self.call_zpc('relocate_container__v',
                      self.ptr, memsrc, c_int8(proc_id))

    def resize(self, size: int):
        self._size = size
        self.call_zpc('resize_container__v', self.ptr, c_size_t(size))

    def get_val(self):
        return self.call_zpc('get_val_container__v', self.ptr)

    def set_val(self, new_val):
        return self.call_zpc(
            'set_val_container__v', self.ptr, self.elem_type.c_type(new_val))


class Vector(VectorAPI):
    def __init__(
            self, memsrc, size, elem_type=fl, dev_id=0, virtual=False) -> None:
        c_memsrc = MemSrc.from_name(memsrc)
        allocator = zpc_lib.lib.allocator(c_memsrc, dev_id)
        api_suffix = f'{elem_type.name}' + ('_virtual' if virtual else '')
        ptr = zpc_lib.call(
            f'container__v_{api_suffix}', allocator, c_size_t(size))
        super().__init__(ptr, memsrc, size, elem_type, dev_id, virtual)

    def __del__(self):
        self.call_zpc(f'del_container__v', self.ptr)


def create_vector_object(memsrc, size, elem_type, dev_id, virtual):
    _handle = c_uint64()
    c_memsrc = MemSrc.from_name(memsrc)
    allocator = zpc_lib.lib.allocator(c_memsrc, dev_id)
    api_suffix = f'{elem_type.name}' + ('_virtual' if virtual else '')
    zeno_lib.call(f'container_obj__v_{api_suffix}', byref(
        _handle), allocator, c_size_t(size))
    return _handle.value


def vector_info_from_handle(handle):
    _ptr = ctypes.c_void_p()
    _memsrc = ctypes.c_int()
    _size = ctypes.c_size_t()
    _dtype = ctypes.c_int()
    _proc_id = ctypes.c_int8()
    _is_virtual = ctypes.c_bool()
    zeno_lib.call('ZS_GetVectorData', c_uint64(handle), byref(_ptr), byref(_memsrc),
                  byref(_size), byref(_dtype), byref(_proc_id), byref(_is_virtual))
    return _ptr.value, MemSrc.to_name(_memsrc.value), _size.value, \
        ind2zstype[_dtype.value], _proc_id.value, _is_virtual.value


class VectorObject(VectorAPI, ZenoObject):
    def __init__(self, memsrc, size, elem_type=fl, dev_id=0, virtual=False,
                 handle=None, ptr=None) -> None:
        if handle is None:
            handle = create_vector_object(
                memsrc, size, elem_type, dev_id, virtual)
        if ptr is None:
            ptr, _, _, _, _, _ = vector_info_from_handle(handle)
        VectorAPI.__init__(self, ptr, memsrc, size, elem_type, dev_id, virtual)
        ZenoObject.__init__(self, handle)

    @staticmethod
    def from_handle(handle):
        ptr, memsrc, size, elem_type, dev_id, virtual = vector_info_from_handle(
            handle)
        return VectorObject(memsrc, size, elem_type, dev_id, virtual, handle, ptr)

    def __del__(self):
        ZenoObject.__del__(self)


# TODO: indicing, rows, cols, ...
# TODO: data_ptr for SmallVecObject
class SmallVecAPI:
    def __init__(self, ptr, shape=(1,), dtype: DataType = fl, data_ptr=None) -> None:
        shape_suffix = '_' + '_'.join(str(x)
                                      for x in shape) if len(shape) else ''
        self.api_suffix = dtype.name + shape_suffix
        self.ptr = ptr
        self.data_ptr = data_ptr
        self.shape = shape
        self.dtype = dtype

    @staticmethod
    def from_numpy(arr: np.ndarray):
        raise NotImplementedError()

    @property
    def size(self):
        return 1 if not len(self.shape) \
            else reduce(lambda a, b: a * b, self.shape)

    def to_numpy(self):
        sizeinbytes = self.size * ctypes.sizeof(self.dtype.c_type)
        np_dtype = zstype2nptype[self.dtype]
        arr = np.empty(self.shape, np_dtype)
        arr = np.ascontiguousarray(arr)
        assert sizeinbytes == arr.size * arr.dtype.itemsize
        ctypes.memmove(arr.ctypes.data, self.data_ptr, sizeinbytes)
        return arr.reshape(self.shape)

    def call_zpc(self, func_name, *args):
        return zpc_lib.call(f'{func_name}_{self.api_suffix}', *args)


class SmallVec(SmallVecAPI):
    def __init__(self, shape=(1,), dtype: DataType = fl) -> None:
        shape_suffix = '_' + '_'.join(str(x)
                                      for x in shape) if len(shape) else ''
        api_suffix = dtype.name + shape_suffix
        ptr = zpc_lib.call('small_vec__' + api_suffix)
        data_ptr = zpc_lib.call('small_vec_data_ptr__' + api_suffix, ptr)
        super().__init__(ptr, shape, dtype, data_ptr)

    @staticmethod
    def from_numpy(arr: np.ndarray):
        assert (len(arr.shape) == 2)
        for dim in arr.shape:
            assert (dim <= 4)
        zs_dtype = nptype2zstype[arr.dtype]
        small_vec = SmallVec(arr.shape, zs_dtype)
        sizeinbytes = small_vec.size * ctypes.sizeof(small_vec.dtype.c_type)
        arr = np.ascontiguousarray(arr)
        assert sizeinbytes == arr.size * arr.dtype.itemsize
        ctypes.memmove(small_vec.ptr, arr.ctypes.data, sizeinbytes)
        return small_vec

    def __del__(self):
        self.call_zpc('del_small_vec_', self.ptr)


def create_svec_obj(shape, dtype: DataType):
    _handle = c_uint64()
    dtype_str = ctype2str[dtype.c_type]
    if len(shape) == 0:
        zeno_lib.call(
            f'ZS_CreateObjectZsSmallVec_{dtype_str}_scalar', byref(_handle))
    elif len(shape) == 1:
        zeno_lib.call(
            f'ZS_CreateObjectZsSmallVec_{dtype_str}_{shape[0]}', byref(_handle))
    elif len(shape) == 2:
        zeno_lib.call(
            f'ZS_CreateObjectZsSmallVec_{dtype_str}_{shape[0]}x{shape[1]}',
            byref(_handle))
    return _handle.value


def svec_info_from_handle(handle):
    _dims = ctypes.c_size_t()
    _ptr = ctypes.c_void_p()
    _data_ptr = ctypes.c_void_p()
    _dim_x = ctypes.c_size_t()
    _dim_y = ctypes.c_size_t()
    _type_ind = ctypes.c_int()
    zeno_lib.call(
        'ZS_GetObjectZsVecData', ctypes.c_uint64(handle),
        ctypes.byref(_ptr),
        ctypes.byref(_dims),
        ctypes.byref(_dim_x),
        ctypes.byref(_dim_y),
        ctypes.byref(_type_ind), 
        ctypes.byref(_data_ptr))
    shape = ()
    if _dims.value == 0:
        shape = ()
    elif _dims.value == 1:
        shape = (_dim_x.value, )
    elif _dims.value == 2:
        shape = (_dim_x.value, _dim_y.value)
    else:
        raise Exception(f"unsupported dim {_dims.value}")
    return _ptr.value, _data_ptr.value, shape, ind2zstype[_type_ind.value]


class SmallVecObject(SmallVecAPI, ZenoObject):
    def __init__(self, shape=(1, ), dtype: DataType = fl, handle=None, ptr=None) -> None:
        if handle is None:
            handle = create_svec_obj(shape, dtype)
        if ptr is None:
            ptr, data_ptr, _, _ = svec_info_from_handle(handle)
        SmallVecAPI.__init__(self, ptr, shape, dtype, data_ptr)
        ZenoObject.__init__(self, handle)

    @staticmethod
    def from_handle(handle):
        ptr, data_ptr, shape, dtype = svec_info_from_handle(handle)
        return SmallVecObject(shape, dtype, handle, ptr, data_ptr)

    @staticmethod
    def from_numpy(arr: np.ndarray):
        assert (len(arr.shape) == 2)
        for dim in arr.shape:
            assert (dim <= 4)
        zs_dtype = nptype2zstype[arr.dtype]
        small_vec_obj = SmallVecObject(arr.shape, zs_dtype)
        sizeinbytes = small_vec_obj.size * \
            ctypes.sizeof(small_vec_obj.dtype.c_type)
        arr = np.ascontiguousarray(arr)
        assert sizeinbytes == arr.size * arr.dtype.itemsize
        ctypes.memmove(small_vec_obj.ptr, arr.ctypes.data, sizeinbytes)
        return small_vec_obj

    def __del__(self):
        ZenoObject.__del__(self)


class TileVectorViewType(ViewType):
    def __init__(self, elem_type, length, with_tile) -> None:
        self.elem_type = elem_type
        self.length = length
        self.with_tile = with_tile
        with_tile = 'true' if with_tile else 'false'
        super().__init__(
            f'zs::TileVectorViewLite<{elem_type.name}, {length}, {with_tile}>')


class TileVectorNamedViewType(ViewType):
    def __init__(self, elem_type, length, with_tile) -> None:
        self.elem_type = elem_type
        self.length = length
        self.with_tile = with_tile
        with_tile = 'true' if with_tile else 'false'
        super().__init__(
            f'zs::TileVectorNamedViewLite<{elem_type.name}, {length}, {with_tile}>')


class VectorViewType(ViewType):
    def __init__(self, elem_type) -> None:
        self.elem_type = elem_type
        super().__init__(f'zs::VectorViewLite<{elem_type.name}>')


def make_tile_vector_view_type(elem_type, length=32, with_tile=False):
    return TileVectorViewType(elem_type, length, with_tile)


def make_tile_vector_named_view_type(elem_type, length=32, with_tile=False):
    return TileVectorNamedViewType(elem_type, length, with_tile)


def make_vector_view_type(elem_type):
    return VectorViewType(elem_type)


tvv_t = make_tile_vector_view_type
tvnv_t = make_tile_vector_named_view_type
vv_t = make_vector_view_type


class TileVectorType(ContainerType):
    def __init__(self, elem_type, length, virtual) -> None:
        self.elem_type = elem_type
        self.length = length
        self.virtual = virtual
        is_virtual = 'true' if virtual else 'false'
        super().__init__(
            f'zs::TileVector<{elem_type.name}, {length}, zs::ZSPmrAllocator<{is_virtual}>>>')

    def __call__(self, memsrc, tags, size) -> TileVector:
        return TileVector(
            memsrc, tags, size, elem_type=self.elem_type, length=self.length,
            virtual=self.virtual)

    def view_t(self, named=True, with_tile=False):
        if named:
            return tvnv_t(self.elem_type, self.length, with_tile=with_tile)
        else:
            return tvv_t(self.elem_type, self.length, with_tile=with_tile)


class VectorType(ContainerType):
    def __init__(self, elem_type, virtual) -> None:
        self.elem_type = elem_type
        self.virtual = virtual
        is_virtual = 'true' if virtual else 'false'
        super().__init__(
            f'zs::Vector<{elem_type.name}, zs::ZSPmrAllocator<{is_virtual}>>')

    def __call__(self, memsrc, size, dev_id=0) -> Vector:
        return Vector(memsrc, size, self.elem_type, dev_id, self.virtual)

    def view_t(self):
        return vv_t(self.elem_type)


class SmallVectorType(ContainerType):
    def __init__(self, shape, dtype) -> None:
        type_name = None
        if not len(shape):
            type_name = ''
        else:
            type_name = f'zs::vec<{dtype.name}, ' + \
                ','.join(str(x) for x in shape) + '>'
        DataType.__init__(self, type_name, in_kernel=True)

    def __call__(self, shape, dtype) -> SmallVec:
        return SmallVec(shape, dtype)


def make_tile_vector_type(elem_type, length=32, virtual=False):
    return TileVectorType(elem_type, length, virtual)


def make_vector_type(elem_type, virtual=False):
    return VectorType(elem_type, virtual)


def make_small_vector_type(shape, dtype):
    return SmallVectorType(shape, dtype)


tv_t = make_tile_vector_type
v_t = make_vector_type
svec_t = make_small_vector_type
svec = SmallVec
