// RUN: %clangxx -fsycl -fsycl-device-only -S -emit-llvm -x c++ %s -o - | FileCheck %s

// This test checks that arithmetic opration on simd_view and scalar does not
// truncate scalar to the view's element type.

#include <sycl/ext/intel/experimental/esimd.hpp>

using namespace sycl::ext::intel::experimental::esimd;

simd<short, 4> test1(int X, simd<short, 16> Y) __attribute__((sycl_device)) {
  // CHECK-LABEL: test1
  // CHECK-SAME: i32 [[X:%.+]],
  // CHECK-NOT: trunc i32 [[X]] to i16
  return Y.select<4, 1>(0) * X;
}

simd<short, 4> test2(int X, simd<short, 16> Y) __attribute__((sycl_device)) {
  // CHECK-LABEL: test2
  // CHECK-SAME: i32 [[X:%.+]],
  // CHECK-NOT: trunc i32 [[X]] to i16
  return Y.select<4, 1>(0) * X;
}
