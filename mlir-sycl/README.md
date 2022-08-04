# SYCL MLIR dialect

## Introduction

MLIR-SYCL is an MLIR dialect to represent SYCL enties.

The project is at a very early stage and work in progress.
For now the dialect can only represent in-kernel SYCL standard object such as id, range, accessor etc.

The project is published under the Apache 2.0 with LLVM Exceptions license.

## Pre-requisites

* CMake (minimum 3.13.4)
* Ninja (optional)
* LLVM / MLIR build or sources
  * Implies LLVM / MLIR (pre-requisites)[https://llvm.org/docs/CMake.html#quick-start]

## Build instructions

To build as an out-of-tree project:

```
mkdir build
cd build
cmake -G Ninja <PATH_TO_MLIR_SYCL> -DMLIR_DIR=<PATH_TO_YOUR_LLVM_REPO>/build/lib/cmake/mlir
ninja
```

The project can also be built as part of the general llvm project.
In this case you need to build the project with mlir enabled.

The configuration of the llvm project should look like:

```
mkdir build
cd build
cmake -G Ninja <PATH_TO_LLVM> -DLLVM_ENABLE_PROJECTS=mlir;mlir-sycl -DLLVM_EXTERNAL_PROJECTS=mlir-sycl -DLLVM_EXTERNAL_MLIR_SYCL_SOURCE_DIR=<PATH_TO_MLIR_SYCL>
ninja
```

## Contributions

This project is licensed under the Apache 2.0 with LLVM Exceptions license.
Contributions are very welcome!
