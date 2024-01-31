// RUN: %clangxx -fsycl -fsycl-device-only -S %s -o - | FileCheck %s

// Check efficiency of LLVM IR generated for various simd constructors.

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

// clang-format off

// Array-based constructor, FP element type, no loops exected - check.
SYCL_EXTERNAL auto foo(double i) SYCL_ESIMD_FUNCTION {
// CHECK: define dso_local spir_func void @_Z3food(
//   CHECK: {{[^,]*}} %[[RES:[a-zA-Z0-9_\.]+]],
//   CHECK: {{[^,]*}} %[[I:[a-zA-Z0-9_\.]+]]){{.*}} {
  simd<double, 2> val({ i, i });
  return val;
// CHECK: %[[V0:[a-zA-Z0-9_\.]+]] = insertelement <2 x double> poison, double %[[I]], i64 0
// CHECK-NEXT: %[[V1:[a-zA-Z0-9_\.]+]] = shufflevector <2 x double> %[[V0]], <2 x double> poison, <2 x i32> zeroinitializer
// CHECK-NEXT: store <2 x double> %[[V1]], ptr addrspace(4) %[[RES]]
// CHECK-NEXT: ret void
// CHECK-NEXT: }
}

// Const base + step constructor, FP element type.
SYCL_EXTERNAL auto double_base_step_const() SYCL_ESIMD_FUNCTION {
  // CHECK: define dso_local spir_func void @_Z22double_base_step_constv({{.*}} %[[RES:[a-zA-Z0-9_\.]+]]){{.*}} {
  return simd<double, 64>{1.0, 3.0};
  // CHECK: store <64 x double> <double 1.000000e+00, double 4.000000e+00, double 7.000000e+00, double 1.000000e+01, double 1.300000e+01, double 1.600000e+01, double 1.900000e+01, double 2.200000e+01, double 2.500000e+01, double 2.800000e+01, double 3.100000e+01, double 3.400000e+01, double 3.700000e+01, double 4.000000e+01, double 4.300000e+01, double 4.600000e+01, double 4.900000e+01, double 5.200000e+01, double 5.500000e+01, double 5.800000e+01, double 6.100000e+01, double 6.400000e+01, double 6.700000e+01, double 7.000000e+01, double 7.300000e+01, double 7.600000e+01, double 7.900000e+01, double 8.200000e+01, double 8.500000e+01, double 8.800000e+01, double 9.100000e+01, double 9.400000e+01, double 9.700000e+01, double 1.000000e+02, double 1.030000e+02, double 1.060000e+02, double 1.090000e+02, double 1.120000e+02, double 1.150000e+02, double 1.180000e+02, double 1.210000e+02, double 1.240000e+02, double 1.270000e+02, double 1.300000e+02, double 1.330000e+02, double 1.360000e+02, double 1.390000e+02, double 1.420000e+02, double 1.450000e+02, double 1.480000e+02, double 1.510000e+02, double 1.540000e+02, double 1.570000e+02, double 1.600000e+02, double 1.630000e+02, double 1.660000e+02, double 1.690000e+02, double 1.720000e+02, double 1.750000e+02, double 1.780000e+02, double 1.810000e+02, double 1.840000e+02, double 1.870000e+02, double 1.900000e+02>, ptr addrspace(4) %[[RES]]
  // CHECK-NEXT: ret void
}

// Variable base + step constructor, FP element type.
SYCL_EXTERNAL auto double_base_step_var(double base, double step) SYCL_ESIMD_FUNCTION {
  // CHECK: define dso_local spir_func void @_Z20double_base_step_vardd({{.*}} %[[RES:[a-zA-Z0-9_\.]+]], double noundef %[[BASE:[a-zA-Z0-9_\.]+]], double noundef %[[STEP:[a-zA-Z0-9_\.]+]]){{.*}} {
  return simd<double, 32>{base, step};
  // CHECK: %[[BASE_VEC_TMP:[a-zA-Z0-9_\.]+]] = insertelement <32 x double> poison, double %[[BASE]], i64 0
  // CHECK: %[[BASE_VEC:[a-zA-Z0-9_\.]+]] = shufflevector <32 x double> %[[BASE_VEC_TMP]], <32 x double> poison, <32 x i32> zeroinitializer
  // CHECK: %[[STEP_VEC_TMP:[a-zA-Z0-9_\.]+]] = insertelement <32 x double> poison, double %[[STEP]], i64 0
  // CHECK: %[[STEP_VEC:[a-zA-Z0-9_\.]+]] = shufflevector <32 x double> %[[STEP_VEC_TMP]], <32 x double> poison, <32 x i32> zeroinitializer
  // CHECK: %[[FMA_VEC:[a-zA-Z0-9_\.]+]] = tail call noundef <32 x double> @llvm.fmuladd.v32f64(<32 x double> %[[STEP_VEC]], <32 x double> <double 0.000000e+00, double 1.000000e+00, double 2.000000e+00, double 3.000000e+00, double 4.000000e+00, double 5.000000e+00, double 6.000000e+00, double 7.000000e+00, double 8.000000e+00, double 9.000000e+00, double 1.000000e+01, double 1.100000e+01, double 1.200000e+01, double 1.300000e+01, double 1.400000e+01, double 1.500000e+01, double 1.600000e+01, double 1.700000e+01, double 1.800000e+01, double 1.900000e+01, double 2.000000e+01, double 2.100000e+01, double 2.200000e+01, double 2.300000e+01, double 2.400000e+01, double 2.500000e+01, double 2.600000e+01, double 2.700000e+01, double 2.800000e+01, double 2.900000e+01, double 3.000000e+01, double 3.100000e+01>, <32 x double> %[[BASE_VEC]])
  // CHECK: store <32 x double> %[[FMA_VEC]], ptr addrspace(4) %[[RES]]
  // CHECK-NEXT: ret void
}

// Const base + step constructor, integer element type.
SYCL_EXTERNAL auto int_base_step_const() SYCL_ESIMD_FUNCTION {
  // CHECK: define dso_local spir_func void @_Z19int_base_step_constv({{.*}} %[[RES:[a-zA-Z0-9_\.]+]]){{.*}} {
  simd<int, 16> val(17, 3);
  return val;
  // CHECK: store <16 x i32> <i32 17, i32 20, i32 23, i32 26, i32 29, i32 32, i32 35, i32 38, i32 41, i32 44, i32 47, i32 50, i32 53, i32 56, i32 59, i32 62>, ptr addrspace(4) %[[RES]]
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
}

// Variable base + step constructor, integer element type.
SYCL_EXTERNAL auto int_base_step_var(int base, int step) SYCL_ESIMD_FUNCTION {
  // CHECK: define dso_local spir_func void @_Z17int_base_step_varii({{.*}} %[[RES:[a-zA-Z0-9_\.]+]], i32 noundef %[[BASE:[a-zA-Z0-9_\.]+]], i32 noundef %[[STEP:[a-zA-Z0-9_\.]+]]){{.*}} {
  return simd<int, 32>{base, step};
  // CHECK: %[[BASE_VEC_TMP:[a-zA-Z0-9_\.]+]] = insertelement <32 x i32> poison, i32 %[[BASE]], i64 0
  // CHECK: %[[BASE_VEC:[a-zA-Z0-9_\.]+]] = shufflevector <32 x i32> %[[BASE_VEC_TMP]], <32 x i32> poison, <32 x i32> zeroinitializer
  // CHECK: %[[STEP_VEC_TMP:[a-zA-Z0-9_\.]+]] = insertelement <32 x i32> poison, i32 %[[STEP]], i64 0
  // CHECK: %[[STEP_VEC:[a-zA-Z0-9_\.]+]] = shufflevector <32 x i32> %[[STEP_VEC_TMP]], <32 x i32> poison, <32 x i32> zeroinitializer
  // CHECK: %[[MUL_VEC:[a-zA-Z0-9_\.]+]] = mul <32 x i32> %[[STEP_VEC]], <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  // CHECK: %[[ADD_VEC:[a-zA-Z0-9_\.]+]] = add <32 x i32> %[[BASE_VEC]], %[[MUL_VEC]]
  // CHECK: store <32 x i32> %[[ADD_VEC]], ptr addrspace(4) %[[RES]]
  // CHECK-NEXT: ret void
}

// Broadcast constructor, FP element type, no loops exected - check.
SYCL_EXTERNAL auto gee() SYCL_ESIMD_FUNCTION {
// CHECK: define dso_local spir_func void @_Z3geev({{.*}} %[[RES:[a-zA-Z0-9_\.]+]]){{.*}} {
  simd<float, 2> val(-7);
  return val;
// CHECK: store <2 x float> <float -7.000000e+00, float -7.000000e+00>, ptr addrspace(4) %[[RES]]
// CHECK-NEXT: ret void
// CHECK-NEXT: }
}

// Array-based simd_mask constructor, no loops exected - check.
SYCL_EXTERNAL auto foomask() SYCL_ESIMD_FUNCTION {
// CHECK: define dso_local spir_func void @_Z7foomaskv({{.*}} %[[RES:[a-zA-Z0-9_\.]+]]){{.*}} {
  simd_mask<2> val({ 1, 0 });
  return val;
// CHECK: store <2 x i16> <i16 1, i16 0>, ptr addrspace(4) %[[RES]]
// CHECK-NEXT: ret void
// CHECK-NEXT: }
}

// Broadcast simd_mask constructor, no loops exected - check.
SYCL_EXTERNAL auto geemask() SYCL_ESIMD_FUNCTION {
// CHECK: define dso_local spir_func void @_Z7geemaskv({{.*}} %[[RES:[a-zA-Z0-9_\.]+]]){{.*}} {
  simd_mask<2> val(1);
  return val;
// CHECK: store <2 x i16> <i16 1, i16 1>, ptr addrspace(4) %[[RES]]
// CHECK-NEXT: ret void
// CHECK-NEXT: }
}

// The element type is 'half', which requires conversion, so code generation
// is less efficient - has loop over elements. No much reason to check.
SYCL_EXTERNAL auto foohalf(half i) SYCL_ESIMD_FUNCTION {
  simd<half, 2> val({ i, i });
  return val;
}

// The element type is 'half', which requires conversion, so code generation
// is less efficient - has loop over elements. No much reason to check.
SYCL_EXTERNAL auto barhalf() SYCL_ESIMD_FUNCTION {
  simd<half, 2> val(17, 3);
  return val;
}

// Here the element is half too, but code generation is efficient because
// no per-element operations are needed - scalar is converted before broadcasting.
SYCL_EXTERNAL auto geehalf() SYCL_ESIMD_FUNCTION {
// CHECK: define dso_local spir_func void @_Z7geehalfv({{.*}} %[[RES:[a-zA-Z0-9_\.]+]]){{.*}} {
  simd<half, 2> val(-7);
  return val;
// CHECK: store <2 x half> <half 0xHC700, half 0xHC700>, ptr addrspace(4) %[[RES]]
// CHECK-NEXT: ret void
// CHECK-NEXT: }
}
