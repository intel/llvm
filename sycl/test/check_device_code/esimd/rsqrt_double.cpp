// RUN: %clangxx -fsycl -fsycl-device-only -S -emit-llvm %s -o - | \
// RUN: FileCheck %s --implicit-check-not=__spirv_ocl_rsqrt

// This test checks that all we use SPIR-V rsqrt for doubles.

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

SYCL_ESIMD_KERNEL SYCL_EXTERNAL void kernel() {
  __ESIMD_NS::simd<double, 16> v(0, 1);
  v = __ESIMD_NS::rsqrt(v);
  // CHECK: recip
}
