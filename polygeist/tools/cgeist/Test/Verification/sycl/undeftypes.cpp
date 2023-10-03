// RUN: clang++  -fsycl -fsycl-device-only -fsycl-targets=spir64-unknown-unknown-syclmlir -O0 -w -emit-mlir -o - -Xcgeist -allow-undefined-sycl-types %s | FileCheck %s
// RUN: clang++  -fsycl -fsycl-device-only -fsycl-targets=spir64-unknown-unknown-syclmlir -O0 -w -emit-mlir -o - -Xcgeist -allow-undefined-sycl-types -D DETAIL_NS %s | FileCheck %s
// RUN: not clang++  -fsycl -fsycl-device-only -fsycl-targets=spir64-unknown-unknown-syclmlir -O0 -w -emit-mlir  -o - -Xcgeist -allow-undefined-sycl-types=false %s 2> >(FileCheck %s --check-prefix=CHECK-ERROR)
// RUN: clang++  -fsycl -fsycl-device-only -fsycl-targets=spir64-unknown-unknown-syclmlir -O0 -w -emit-mlir -o - -Xcgeist -allow-undefined-sycl-types=false -D DETAIL_NS %s | FileCheck %s

#include <sycl/sycl.hpp>

template <typename T> SYCL_EXTERNAL void keep(T);

namespace sycl {
#ifdef DETAIL_NS
namespace detail {
#endif  // DETAIL_NS
class VecWrapper {
  float4 v;
public:
  SYCL_EXTERNAL explicit VecWrapper(float);
  SYCL_EXTERNAL float operator[](size_t) const;
};
#ifdef DETAIL_NS
} // namespace detail
#endif  // DETAIL_NS
}  // namespace sycl

using namespace sycl;
#ifdef DETAIL_NS
using namespace sycl::detail;
#endif  // DETAIL_NS

// CHECK-LABEL:  func.func @_Z16vec_wrapper_testfm(
// CHECK-SAME:     %[[SPAN:.*]]: f32
// CHECK-SAME:     %[[INDEX:.*]]: i64
// CHECK:          sycl.call @VecWrapper(%{{.*}}, %[[SPAN]]) {MangledFunctionName = @{{.*}}, TypeName = @VecWrapper} : (memref<?x!llvm.struct<(!sycl_vec_f32_4_)>, 4>, f32) -> ()
// CHECK:          %{{.*}} = sycl.call @"operator[]"(%{{.*}}, %[[INDEX]]) {MangledFunctionName = @{{.*}}, TypeName = @VecWrapper} : (memref<?x!llvm.struct<(!sycl_vec_f32_4_)>, 4>, i64) -> f32

// CHECK-ERROR:  Found type in the sycl namespace, but not in the SYCL dialect

SYCL_EXTERNAL void vec_wrapper_test(float span, size_t i) {
  VecWrapper w{span};
  keep(w[i]);
}
