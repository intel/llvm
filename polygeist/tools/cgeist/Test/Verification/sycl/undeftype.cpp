// RUN: clang++ -fsycl -fsycl-device-only -fsycl-targets=spir64-unknown-unknown-syclmlir -O0 -w -emit-mlir -o - %s | FileCheck %s

#include <sycl/sycl.hpp>

template <typename T> SYCL_EXTERNAL void keep(T);

namespace sycl {
class VecWrapper {
  float4 v;
public:
  SYCL_EXTERNAL explicit VecWrapper(float);
  SYCL_EXTERNAL float operator[](size_t) const;
};
}  // namespace sycl

// CHECK-LABEL:  func.func @_Z16vec_wrapper_testfm(
// CHECK-SAME:     %[[SPAN:.*]]: f32
// CHECK-SAME:     %[[INDEX:.*]]: i64
// CHECK:          sycl.call(%{{.*}}, %[[SPAN]]) {FunctionName = @VecWrapper, MangledFunctionName = @{{.*}}, TypeName = @VecWrapper} : (memref<?x!llvm.struct<(!sycl_vec_f32_4_)>, 4>, f32) -> ()
// CHECK:          %{{.*}} = sycl.call(%{{.*}}, %[[INDEX]]) {FunctionName = @"operator[]", MangledFunctionName = @{{.*}}, TypeName = @VecWrapper} : (memref<?x!llvm.struct<(!sycl_vec_f32_4_)>, 4>, i64) -> f32

SYCL_EXTERNAL void vec_wrapper_test(float span, size_t i) {
  sycl::VecWrapper w{span};
  keep(w[i]);
}
