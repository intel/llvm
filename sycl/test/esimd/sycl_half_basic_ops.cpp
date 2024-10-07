// RUN: %clangxx -fsycl -fsycl-device-only -S %s -o %t
// RUN: sycl-post-link -properties -split-esimd -lower-esimd -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll

// The test checks that there are no unexpected extra conversions or intrinsic
// calls added by the API headers or compiler when generating code
// for basic C++ operations on simd<sycl::half, N> values.

#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel;
using namespace sycl;

// clang-format off
// --- Unary operation
SYCL_EXTERNAL auto test_unary_op(simd<sycl::half, 8> val) SYCL_ESIMD_FUNCTION {
// CHECK: define dso_local spir_func <8 x half> @_Z13test_unary_op{{.*}}(
//   CHECK: {{.*}} %[[VAL_VEC:[a-zA-Z0-9_\.]+]]){{.*}} {
  return -val;
// CHECK: %[[RES:[a-zA -Z0-9_\.]+]] = fneg <8 x half> %[[VAL_VEC]]
// CHECK-NEXT: ret <8 x half> %[[RES]]
// CHECK-LABEL: }
}

// --- Binary operation on <half, half> pair
SYCL_EXTERNAL auto test_binary_op1(simd<sycl::half, 8> val1, simd<sycl::half, 8> val2) SYCL_ESIMD_FUNCTION {
// CHECK: define dso_local spir_func <8 x half> @_Z15test_binary_op1{{.*}}(
//   CHECK: {{[^,]*}} %[[VAL1_VEC:[a-zA-Z0-9_\.]+]],
//   CHECK: %[[VAL2_VEC:[a-zA-Z0-9_\.]+]]{{.*}} {
  return val1 + val2;
// CHECK: %[[RES:[a-zA -Z0-9_\.]+]] = fadd <8 x half> %[[VAL1_VEC]], %[[VAL2_VEC]]
// CHECK-NEXT: ret <8 x half> %[[RES]]
// CHECK-LABEL: }
}

// --- Binary operation on <half, int64_t> pair
// The integer operand is expected to be converted to half type.
SYCL_EXTERNAL auto test_binary_op2(simd<sycl::half, 8> val1, simd<long long, 8> val2) SYCL_ESIMD_FUNCTION {
// CHECK: define dso_local spir_func <8 x half> @_Z15test_binary_op2{{[^\(]*}}(
//   CHECK: <8 x half> %[[VAL1_VEC:[a-zA-Z0-9_\.]+]],
//   CHECK: <8 x i64> %[[VAL2_VEC:[a-zA-Z0-9_\.]+]]{{.*}} {
  return val1 + val2;
// CHECK: %[[CONV:[a-zA-Z0-9_\.]+]] = sitofp <8 x i64> %[[VAL2_VEC]] to <8 x half>
// CHECK-NEXT: %[[RES:[a-zA-Z0-9_\.]+]] = fadd <8 x half> %[[CONV]], %[[VAL1_VEC]]
// CHECK-NEXT: ret <8 x half> %[[RES]]
// CHECK-LABEL: }
}

// --- Comparison operation on <half, int64_t> pair
// The integer operand is expected to be converted to half type.
SYCL_EXTERNAL auto test_cmp_op(simd<sycl::half, 8> val1, simd<long long, 8> val2) SYCL_ESIMD_FUNCTION {
// CHECK: define dso_local spir_func <8 x i16> @_Z11test_cmp_op{{[^\(]*}}(
//   CHECK: <8 x half> %[[VAL1_VEC:[a-zA-Z0-9_\.]+]],
//   CHECK: <8 x i64> %[[VAL2_VEC:[a-zA-Z0-9_\.]+]]{{.*}} {
  return val1 < val2;
// CHECK: %[[CONV:[a-zA-Z0-9_\.]+]] = sitofp <8 x i64> %[[VAL2_VEC]] to <8 x half>
// CHECK-NEXT: %[[CMP:[a-zA-Z0-9_\.]+]] = fcmp ogt <8 x half> %[[CONV]], %[[VAL1_VEC]]
// CHECK-NEXT: %[[RES:[a-zA-Z0-9_\.]+]] = zext <8 x i1> %[[CMP]] to <8 x i16>
// CHECK-NEXT: ret <8 x i16>{{.*}}%[[RES]]
// CHECK-LABEL: }
}
// clang-format on
