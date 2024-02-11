import os
import sys
from glob import glob 
import pathlib
import subprocess
import shutil
from os.path import dirname, join as pjoin
from setuptools import setup, find_packages


def find_file_dir_recursive(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return root


meta = {}
with open(pjoin('pyzpc', '__version__.py')) as f:
    exec(f.read(), meta)

cwd = pathlib.Path().absolute()
base_dir = dirname(__file__)
cmake_dir = os.path.join(base_dir, 'pyzpc', 'zpc_jit')
os.chdir(cmake_dir)
build_dir = pjoin(base_dir, 'cmake_build')
out_lib_dir = pjoin(cmake_dir, 'lib')
out_lib_dir_path = pathlib.Path(out_lib_dir)
out_lib_dir_path.mkdir(parents=True, exist_ok=True)
cmake_args = [
    f'-B{build_dir}',
    '-DCMAKE_BUILD_TYPE=Release',
    '-DZS_ENABLE_ZENO_CU_WRANGLE=OFF',
    '-DZS_ENABLE_VULKAN=OFF',
    '-DZS_ENABLE_JIT=ON',
    '-DZS_ENABLE_CUDA=ON',
    # '-DCMAKE_TOOLCHAIN_FILE=C:/Develop/vcpkg/scripts/buildsystems/vcpkg.cmake',
    '-DZS_BUILD_SHARED_LIBS=ON'
]
build_args = [
    '--config', 'Release',
    '--parallel', str(os.cpu_count())
]
subprocess.run(
    ['cmake', cmake_dir, *cmake_args], cwd='.', check=True
)
subprocess.run(
    ['cmake', '--build', build_dir, *build_args], cwd='.', check=True
)

lib_prefix = '' if os.name == 'nt' else 'lib'
lib_suffix = 'so'
if sys.platform == 'win32':
    lib_suffix = 'dll'
elif sys.platform == 'darwin':
    lib_suffix = 'dylib'
loc_lib_name = f'{lib_prefix}zpc_py_interop.{lib_suffix}'
build_lib_dir = find_file_dir_recursive(loc_lib_name, build_dir)
shared_lib_paths = glob(pjoin(build_dir, '**/*.so'), recursive=True) + \
    glob(pjoin(build_dir, '**/*.dll'), recursive=True) + \
    glob(pjoin(build_dir, '**/*.dylib'), recursive=True)
for path in shared_lib_paths:
    filename = os.path.basename(path)
    shutil.copy(
        path, 
        pjoin(out_lib_dir, filename)
    )
# os.removedirs(build_dir)
os.chdir(str(cwd))

setup(
    name=meta['__title__'],
    version=meta['__version__'],
    url=meta['__url__'],
    description=meta['__description__'],
    long_description=(
        'More information could be found at https://github.com/littlemine/POC_pyzpc'),
    platforms=['any'],
    packages=find_packages(),
    package_data={
        "": ['zpc_jit/zpc/**/*.hpp',
             'zpc_jit/zpc/**/*.h',
             'zpc_jit/zpc/**/*.cuh',
             'zpc_jit/zpc/**/*.cpp',
             'zpc_jit/zpc/**/*.cu',
             'zpc_jit/lib/*.so',
             'zpc_jit/lib/*.dll']
    },
    include_package_data=True,
    classifiers=[],
    tests_require=[],
    install_requires=[
        'numpy',
        'sympy'
    ],
    python_requires='>=3.8'
)
