// RUN: %clangxx -fsycl -fsycl-device-only -S %s -o - | FileCheck %s

// Check efficiency of LLVM IR generated for various simd constructors.

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>

using namespace sycl;
using namespace sycl::ext::intel::experimental::esimd;

// clang-format off

// Array-based constructor, FP element type, no loops exected - check.
SYCL_EXTERNAL auto foo(double i) SYCL_ESIMD_FUNCTION {
// CHECK: define dso_local spir_func void @_Z3food(
//   CHECK: {{[^,]*}} %[[RES:[a-zA-Z0-9_\.]+]],
//   CHECK: {{[^,]*}} %[[I:[a-zA-Z0-9_\.]+]]){{.*}} {
  simd<double, 2> val({ i, i });
  return val;
// CHECK: %[[V0:[a-zA-Z0-9_\.]+]] = insertelement <2 x double> undef, double %[[I]], i64 0
// CHECK-NEXT: %[[V1:[a-zA-Z0-9_\.]+]] = shufflevector <2 x double> %[[V0]], <2 x double> poison, <2 x i32> zeroinitializer
// CHECK-NEXT: %[[MDATA:[a-zA-Z0-9_\.]+]] = getelementptr inbounds {{.*}} %[[RES]], i64 0, i32 0, i32 0
// CHECK-NEXT: store <2 x double> %[[V1]], <2 x double> addrspace(4)* %[[MDATA]]
// CHECK-NEXT: ret void
// CHECK-NEXT: }
}

// Base + step constructor, FP element type, loops exected - don't check.
SYCL_EXTERNAL auto bar() SYCL_ESIMD_FUNCTION {
  simd<double, 2> val(17, 3);
  return val;
}

// Base + step constructor, integer element type, no loops exected - check.
SYCL_EXTERNAL auto baz() SYCL_ESIMD_FUNCTION {
  // CHECK: define dso_local spir_func void @_Z3bazv({{.*}} %[[RES:[a-zA-Z0-9_\.]+]]){{.*}} {
  simd<int, 2> val(17, 3);
  return val;
  // CHECK: %[[MDATA:[a-zA-Z0-9_\.]+]] = getelementptr inbounds {{.*}} %[[RES]], i64 0, i32 0, i32 0
  // CHECK-NEXT: store <2 x i32> <i32 17, i32 20>, <2 x i32> addrspace(4)* %[[MDATA]]
  // CHECK-NEXT: ret void
  // CHECK-NEXT: }
}

// Broadcast constructor, FP element type, no loops exected - check.
SYCL_EXTERNAL auto gee() SYCL_ESIMD_FUNCTION {
// CHECK: define dso_local spir_func void @_Z3geev({{.*}} %[[RES:[a-zA-Z0-9_\.]+]]){{.*}} {
  simd<float, 2> val(-7);
  return val;
// CHECK: %[[MDATA:[a-zA-Z0-9_\.]+]] = getelementptr inbounds {{.*}} %[[RES]], i64 0, i32 0, i32 0
// CHECK-NEXT: store <2 x float> <float -7.000000e+00, float -7.000000e+00>, <2 x float> addrspace(4)* %[[MDATA]]
// CHECK-NEXT: ret void
// CHECK-NEXT: }
}

// Array-based simd_mask constructor, no loops exected - check.
SYCL_EXTERNAL auto foomask() SYCL_ESIMD_FUNCTION {
// CHECK: define dso_local spir_func void @_Z7foomaskv({{.*}} %[[RES:[a-zA-Z0-9_\.]+]]){{.*}} {
  simd_mask<2> val({ 1, 0 });
  return val;
// CHECK: %[[MDATA:[a-zA-Z0-9_\.]+]] = getelementptr inbounds {{.*}} %[[RES]], i64 0, i32 0, i32 0
// CHECK-NEXT: store <2 x i16> <i16 1, i16 0>, <2 x i16> addrspace(4)* %[[MDATA]]
// CHECK-NEXT: ret void
// CHECK-NEXT: }
}

// Broadcast simd_mask constructor, no loops exected - check.
SYCL_EXTERNAL auto geemask() SYCL_ESIMD_FUNCTION {
// CHECK: define dso_local spir_func void @_Z7geemaskv({{.*}} %[[RES:[a-zA-Z0-9_\.]+]]){{.*}} {
  simd_mask<2> val(1);
  return val;
// CHECK: %[[MDATA:[a-zA-Z0-9_\.]+]] = getelementptr inbounds {{.*}} %[[RES]], i64 0, i32 0, i32 0
// CHECK-NEXT: store <2 x i16> <i16 1, i16 1>, <2 x i16> addrspace(4)* %[[MDATA]]
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
// CHECK: %[[MDATA:[a-zA-Z0-9_\.]+]] = getelementptr inbounds {{.*}} %[[RES]], i64 0, i32 0, i32 0
// CHECK-NEXT: store <2 x half> <half 0xHC700, half 0xHC700>, <2 x half> addrspace(4)* %[[MDATA]]
// CHECK-NEXT: ret void
// CHECK-NEXT: }
}
