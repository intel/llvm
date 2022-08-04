## Build instructions

### Requirements 
- Working C and C++ toolchains(compiler, linker)
- cmake
- make or ninja

### 1. Clone Polygeist
```sh
git clone --recursive https://github.com/llvm/Polygeist
cd Polygeist
```

### 2. Install LLVM, MLIR, Clang, and Polygeist

#### Option 1: Using pre-built LLVM, MLIR, and Clang

Polygeist can be built by providing paths to a pre-built MLIR and Clang toolchain.

1. Build LLVM, MLIR, and Clang:
```sh
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-mlir
```

2. Build Polygeist:
```sh
mkdir build
cd build
cmake -G Ninja .. \
  -DMLIR_DIR=$PWD/../llvm-project/build/lib/cmake/mlir \
  -DCLANG_DIR=$PWD/../llvm-project/build/lib/cmake/clang \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-polygeist-opt && ninja check-cgeist
```

#### Option 2: Using unified LLVM, MLIR, Clang, and Polygeist build

Polygeist can also be built as an external LLVM project using [LLVM_EXTERNAL_PROJECTS](https://llvm.org/docs/CMake.html#llvm-related-variables).

1. Build LLVM, MLIR, Clang, and Polygeist:
```sh
mkdir build
cd build
cmake -G Ninja ../llvm-project/llvm \
  -DLLVM_ENABLE_PROJECTS="clang;mlir;opencl;sycl" \
  -DLLVM_EXTERNAL_PROJECTS="polygeist;opencl;sycl" \
  -DLLVM_EXTERNAL_POLYGEIST_SOURCE_DIR=.. \
  -DLLVM_EXTERNAL_SYCL_SOURCE_DIR=../llvm-project/sycl \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DSYCL_HALIDE_PATH=<path to halide> \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-polygeist-opt && ninja check-cgeist
```

`ninja check-polygeist-opt` runs the tests in `Polygeist/test/polygeist-opt`
`ninja check-cgeist` runs the tests in `Polygeist/tools/cgeist/Test`
