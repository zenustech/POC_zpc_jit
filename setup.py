import os
import pathlib
import subprocess
import shutil
from os.path import dirname, join as pjoin
from setuptools import setup, find_packages

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
    # '-DCMAKE_BUILD_TYPE=Release', 
    '-DZS_ENABLE_VULKAN=OFF', 
    '-DZS_ENABLE_JIT=ON', 
    '-DZS_ENABLE_CUDA=ON', 
    '-DWHEREAMI_BUILD_SHARED_LIBS=ON'
]
build_args = [
    '--config', 'Release',
    '-j', str(os.cpu_count())
]
subprocess.run(
    ['cmake', cmake_dir, *cmake_args], cwd='.', check=True
)
subprocess.run(
    ['cmake', '--build', build_dir, *build_args], cwd='.', check=True
)
for filename in os.listdir(build_dir):
    if filename.endswith('.so') or filename.endswith('.dll'):
        shutil.copy(pjoin(build_dir, filename), pjoin(out_lib_dir, filename))
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
