# Proof-of-concept for ZPC JIT

## Build

This build has been tested on windows 10/11 and ubuntu 20/22 with **llvm 15**, **cuda 11.6+** and **python 3.10**. Other configurations might also work just fine. If there are build issues encountered, your feedback is always appreciated.

### Prerequisites

#### [**zpc@py_zfx**](https://github.com/zenustech/zpc/tree/py_zfx)

Currently *zpc jit* module supports these JIT backends: LLVM (with openmp support if its runtime available) and Nvidia nvrtc. **py_zfx** is the target branch that is assumed to build ZPC JIT pipeline.

#### [**llvm 15+**](https://github.com/llvm/llvm-project/releases/tag/llvmorg-15.0.7)

Under linux system,

```bash
sudo apt install libelf-dev
cmake llvm -Bbuild -DLLVM_USE_CRT_RELEASE=MT -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_BUILD_TYPE:STRING=Release -DLLVM_TARGETS_TO_BUILD:STRING=X86 -DLLVM_ENABLE_PROJECTS="clang;openmp;compiler-rt" -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_ENABLE_PEDANTIC=OFF -DLLVM_ENABLE_PIC=ON
cmake --build build --parallel 16
sudo cmake --build build --parallel 16 --target install
```

Under windows system, it is preferred to acquire llvm through vcpkg.

```powershell
.\vcpkg install llvm:x64-windows
```

If a manual installation is demanded, make sure all **ATL components** have already been installed with visual studio, which is required for building LLVM. Then open terminal "**x64 Native Tools Command Prompt for VS20xx**", run the following build script.

```powershell
cmake llvm -Bbuild -G Ninja -DCMAKE_BUILD_TYPE=RELEASE -DLLVM_TARGETS_TO_BUILD="X86" -DLLVM_ENABLE_PROJECTS="clang;openmp;compiler-rt" -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_ENABLE_PIC=ON
cd build
ninja
ninja install
```

*Ninja* generator should be accessible alongside visual studio.

#### [**nvrtc (cuda toolkit)**](https://developer.nvidia.com/cuda-downloads)

Cuda 11.6+ is required for zpc build, though the newest version is generally preferred.

#### **python 3.11**

Whenever you build a cmake c++ project that depends on the python package (i.e. find_package(Python3)) under windows system, it is preferred to put "-DCMAKE_MODULE_PATH=path/to/vs/cmake/path" during cmake configuration, where the *FindPython3.cmake* file that belongs to your visual studio IDE resides (e.g. "C:/Microsoft_Visual_Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/share/cmake-3.26/Modules"), thus avoiding misuse of python from your vcpkg packages.

### Python Package Installation

```bash
pip install . --verbose
```

### Zeno Installation

(TBD)

## Usage

```python
import pyzpc as zs
import zpy  # if used in zeno, this is required
```

### standalone mode

### zeno mode

## Example
