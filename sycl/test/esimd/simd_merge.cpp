// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <limits>
#include <sycl/ext/intel/esimd.hpp>
#include <utility>

using namespace sycl::ext::intel::esimd;

bool test_simd_merge1() __attribute__((sycl_device)) {
  simd<int, 16> v0 = 1;
  simd<int, 16> v1 = 2;
  simd_mask<16> mask = 0;
  mask.select<4, 4>(0) = 1;
  v0.merge(v1, mask);
  return v0[0] == 2 && v0[4] == 2 && v0[8] == 2 && v0[12] == 2;
}

bool test_simd_merge2() __attribute__((sycl_device)) {
  simd<int, 16> v0 = 1;
  simd<int, 16> v1 = 2;
  simd<int, 16> v2 = 3;
  simd_mask<16> mask = 0;
  mask.select<4, 4>(0) = 1;
  v0.merge(v1, v2, (v1 < v2) & mask);
  return v0[0] == 2 && v0[4] == 2 && v0[8] == 2 && v0[12] == 2 && v0[3] == 3 &&
         v0[7] == 3 && v0[11] == 3 && v0[15] == 3;
}

bool test_simd_merge2d1() __attribute__((sycl_device)) {
  simd<int, 16> v0 = 1;
  simd<int, 16> v1 = 2;
  simd_mask<16> mask = 0;
  mask.select<4, 4>(0) = 1;
  auto v0_2d = v0.bit_cast_view<int, 4, 4>();
  v0_2d.merge(v1, mask);
  return v0[0] == 2 && v0[4] == 2 && v0[8] == 2 && v0[12] == 2;
}

bool test_simd_merge2d2() __attribute__((sycl_device)) {
  simd<int, 16> v0 = 1;
  simd<int, 16> v1 = 2;
  simd<int, 16> v2 = 3;
  simd_mask<16> mask = 0;
  mask.select<4, 4>(0) = 1;
  auto v0_2d = v0.bit_cast_view<int, 4, 4>();
  v0_2d.merge(v1, v2, mask);
  return v0[0] == 2 && v0[4] == 2 && v0[8] == 2 && v0[12] == 2 && v0[3] == 3 &&
         v0[7] == 3 && v0[11] == 3 && v0[15] == 3;
}
