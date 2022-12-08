// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -emit-mlir -o - %s | FileCheck %s

#include <sycl/sycl.hpp>

template <typename T> SYCL_EXTERNAL void keep(T);

// CHECK-LABEL: func.func @_Z13access_marrayim(
// CHECK:         call @{{.*}}(%{{.*}}, %{{.*}}) : (!llvm.ptr<struct<(array<256 x i32>)>, 4>, memref<?xi32, 4>) -> ()
// CHECK:         %{{.*}} = call @{{.*}}(%{{.*}}, %arg1) : (!llvm.ptr<struct<(array<256 x i32>)>, 4>, i64) -> memref<?xi32, 4>

SYCL_EXTERNAL void access_marray(int el, size_t i) {
  sycl::marray<int, 256> marr(el);
  keep(marr[i]);
}
