// RUN: %clangxx -fsycl -fsycl-device-only -S %s -o %t.ll
// RUN: sycl-post-link -split-esimd -lower-esimd -S %t.ll -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll

// The test checks that there are no unexpected extra conversions or intrinsic
// calls added by the API headers or compiler when generating code
// for math operations on simd<sycl::half, N> values.

#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel;
using namespace sycl;

// clang-format off
SYCL_EXTERNAL auto test_ext_math_op(simd<sycl::half, 8> val) SYCL_ESIMD_FUNCTION {
// CHECK: define dso_local spir_func void @_Z16test_ext_math_op{{[^\(]*}}(
//   CHECK: <8 x half>{{[^,]*}}* %[[RET_VEC_ADDR:[a-zA-Z0-9_\.]+]],
//   CHECK: <8 x half>* %[[VAL_PTR:[a-zA-Z0-9_\.]+]]){{.*}} {
  return esimd::cos(val);
// CHECK: %[[VAL_VEC_ADDR:[a-zA-Z0-9_\.]+]] = addrspacecast {{.*}} %[[VAL_PTR]]
// CHECK-NEXT: %[[VAL_VEC:[a-zA-Z0-9_\.]+]] = load <8 x half>{{.*}} %[[VAL_VEC_ADDR]]
// CHECK-NEXT: %[[RES:[a-zA-Z0-9_\.]+]] = call <8 x half> @llvm.genx.cos.v8f16(<8 x half> %[[VAL_VEC]])
// CHECK-NEXT: store <8 x half>{{.*}}%[[RES]], {{.*}}%[[RET_VEC_ADDR]]
// CHECK-NEXT: ret void
// CHECK-LABEL: }
}
// clang-format on
