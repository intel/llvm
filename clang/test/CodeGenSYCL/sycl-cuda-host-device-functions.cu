// RUN: %clang_cc1 -fsycl-is-host -sycl-std=2020 -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-HOST
// RUN: %clang_cc1 -fsycl-is-device -sycl-std=2020 -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-DEV

// This tests
// - if a dummy __host__ function (returning undef) is generated for every
//   __device__ function without a host counterpart in sycl-host compilation;
// - if __host__ or __device__  functions are generated correctly depending
//   on the compilation;
// - if __host__ and __device__ functions in sycl-device compilation are
//   generated with weak_odr linkage for avoiding multiple declaration of
//   the same function due to the cuda-device compilation.

#include "../CodeGenCUDA/Inputs/cuda.h"

__host__ int fun0() { return 0; }
__device__ int fun0();

// CHECK-HOST: define dso_local noundef i32 @_Z4fun0v()
// CHECK-HOST: ret i32 0

// CHECK-DEV: declare noundef i32 @_Z4fun0v()

__device__ int fun1() { return 1; }
__host__ int fun1() { return 2; }

// CHECK-HOST: define dso_local noundef i32 @_Z4fun1v()
// CHECK-HOST: ret i32 2

// CHECK-DEV: define weak_odr noundef i32 @_Z4fun1v()
// CHECK-DEV: ret i32 1

__host__ __device__ int fun2() { return 3; }

// CHECK-HOST: define dso_local noundef i32 @_Z4fun2v()
// CHECK-HOST: ret i32 3

// CHECK-DEV: define weak_odr noundef i32 @_Z4fun2v()
// CHECK-DEV: ret i32 3

__host__ int fun3() { return 4; }

// CHECK-HOST: define dso_local noundef i32 @_Z4fun3v()
// CHECK-HOST: ret i32 4

// CHECK-DEV: define weak_odr noundef i32 @_Z4fun3v()
// CHECK-DEV: ret i32 4

__device__ int fun4() { return 6; }
__host__ int fun4();

// CHECK-HOST: declare noundef i32 @_Z4fun4v()

// CHECK-DEV: define weak_odr noundef i32 @_Z4fun4v()
// CHECK-DEV: ret i32 6

__device__ int fun5() { return 5; }

// CHECK-HOST: define dso_local noundef i32 @_Z4fun5v()
// CHECK-HOST: ret i32 undef

// CHECK-DEV: define weak_odr noundef i32 @_Z4fun5v()
// CHECK-DEV: ret i32 5

int fun6() { return 7; }

// CHECK-DEV: define dso_local noundef i32 @_Z4fun6v()
// CHECK-DEV: ret i32 7

__attribute((sycl_device)) void test() {
  fun0();
  fun1();
  fun2();
  fun3();
  fun4();
  fun5();
  fun6();
}

