// The test verifies 16 bit to 4 bit downconversion on Xe3+.

// RUN: %clangxx -fsycl -fsycl-device-only -S %s -o %t.ll
// -O0 lowering, requires `-force-disable-esimd-opt` to disable all
// optimizations.
// RUN: sycl-post-link -split=none -O0 -force-disable-esimd-opt -lower-esimd -split-esimd -S %t.ll -o %t.table
// RUN: FileCheck %s -input-file=%t_0.esimd.ll

#include <sycl/ext/intel/esimd.hpp>

using namespace sycl;
using namespace sycl::ext::intel;
using namespace sycl::ext::intel::esimd;

SYCL_EXTERNAL void func0() SYCL_ESIMD_FUNCTION {
  simd<sycl::half, 8> a = 1;
  simd<sycl::half, 8> b = 2;
  // CHECK: call <4 x i32> @llvm.genx.4bit.downconvert.v4i32(<4 x i32> {{[^,]+}},
  // CHECK-SAME: <4 x i32> {{[^,]+}},
  // CHECK-SAME: <4 x i32> {{[^,]+}},
  // CHECK-SAME: i8 4, i8 3, i8 1)
  simd<uint32_t, 4> res = __ESIMD_ENS::downconvert<
      __ESIMD_ENS::downconvert_packing_mode::mode_3,
      __ESIMD_ENS::downconvert_rounding_mode::round_to_nearest_even,
      __ESIMD_ENS::downconvert_output_mode::s1e2m1>(a, b);
}

SYCL_EXTERNAL void func1() SYCL_ESIMD_FUNCTION {
  simd<sycl::ext::oneapi::bfloat16, 8> a = 1;
  simd<sycl::ext::oneapi::bfloat16, 8> b = 2;
  simd<uint32_t, 4> bias = 3;
  // CHECK: call <4 x i32> @llvm.genx.4bit.downconvert.v4i32(<4 x i32> {{[^,]+}},
  // CHECK-SAME: <4 x i32> {{[^,]+}},
  // CHECK-SAME: <4 x i32> {{[^,]+}},
  // CHECK-SAME: i8 2, i8 3, i8 0)
  simd<uint32_t, 4> res = __ESIMD_ENS::downconvert<
      __ESIMD_ENS::downconvert_packing_mode::mode_3,
      __ESIMD_ENS::downconvert_rounding_mode::biased_round,
      __ESIMD_ENS::downconvert_output_mode::int4>(a, b, bias);
}

SYCL_EXTERNAL void func2() SYCL_ESIMD_FUNCTION {
  simd<sycl::half, 8> a = 1;
  simd<sycl::half, 8> b = 2;
  // CHECK: call <4 x i32> @llvm.genx.4bit.downconvert.v4i32(<4 x i32> {{[^,]+}},
  // CHECK-SAME: <4 x i32> {{[^,]+}},
  // CHECK-SAME: <4 x i32> {{[^,]+}},
  // CHECK-SAME: i8 4, i8 3, i8 1)
  simd<uint32_t, 4> res =
      __ESIMD_ENS::downconvert<__ESIMD_ENS::downconvert_packing_mode::mode_3,
                               __ESIMD_ENS::downconvert_output_mode::s1e2m1>(a,
                                                                             b);
}

SYCL_EXTERNAL void func3() SYCL_ESIMD_FUNCTION {
  simd<sycl::ext::oneapi::bfloat16, 8> a = 1;
  simd<sycl::ext::oneapi::bfloat16, 8> b = 2;
  simd<uint32_t, 4> bias = 3;
  // CHECK: call <4 x i32> @llvm.genx.4bit.downconvert.v4i32(<4 x i32> {{[^,]+}},
  // CHECK-SAME: <4 x i32> {{[^,]+}},
  // CHECK-SAME: <4 x i32> {{[^,]+}},
  // CHECK-SAME: i8 2, i8 3, i8 0)
  simd<uint32_t, 4> res =
      __ESIMD_ENS::downconvert<__ESIMD_ENS::downconvert_packing_mode::mode_3,
                               __ESIMD_ENS::downconvert_output_mode::int4>(
          a, b, bias);
}
