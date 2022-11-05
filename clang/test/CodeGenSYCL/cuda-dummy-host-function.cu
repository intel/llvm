// RUN: %clang_cc1 -triple spir64-unknown-unknown -fsycl-is-host -emit-llvm %s -o - | FileCheck %s

// Test if a dummy __host__ function (returning undef) is generated for every __device__ function without a host counterpart.

#include "../CodeGenCUDA/Inputs/cuda.h"
#include "Inputs/sycl.hpp"

// CHECK: ret i32 2
__device__ int fun0() { return 1; }
__host__ int fun0() { return 2; }

// CHECK: ret i32 3
__host__ __device__ int fun1() { return 3; }

// CHECK: ret i32 undef
__device__ int fun2() { return 4; }
