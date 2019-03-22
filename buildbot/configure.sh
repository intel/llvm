#!/bin/bash

BRANCH=
BUILD_NUMBER=
PR_NUMBER=

# $1 exit code
# $2 error message
exit_if_err()
{
    if [ $1 -ne 0  ]; then
        echo "Error: $2"
        exit $1
    fi
}

unset OPTIND
while getopts ":b:r:n:" option; do
    case $option in
        b) BRANCH=$OPTARG ;;
        n) BUILD_NUMBER=$OPTARG ;;
        r) PR_NUMBER=$OPTARG ;;
    esac
done && shift $(($OPTIND - 1))

# we're in llvm.obj dir
BUILD_DIR=${PWD}

cmake -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS=clang -DLLVM_EXTERNAL_PROJECTS="sycl;llvm-spirv" \
    -DLLVM_EXTERNAL_SYCL_SOURCE_DIR=../llvm.src/sycl -DLLVM_EXTERNAL_LLVM_SPIRV_SOURCE_DIR=../llvm.src/llvm-spirv \
    -DLLVM_TOOL_SYCL_BUILD=ON -DLLVM_TOOL_LLVM_SPIRV_BUILD=ON -DOpenCL_INCLUDE_DIR="OpenCL-Headers" \
    -DOpenCL_LIBRARY="OpenCL-ICD-Loader/build/lib/libOpenCL.so" ../llvm.src/llvm
