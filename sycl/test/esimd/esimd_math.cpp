// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <limits>
#include <sycl/ext/intel/experimental/esimd.hpp>
#include <utility>

using namespace sycl::ext::intel::experimental::esimd;

bool test_esimd_mask() __attribute__((sycl_device)) {
  simd<ushort, 16> a(0);
  a.select<4, 1>(4) = 1;
  a.select<4, 1>(12) = 1;
  unsigned int b = esimd_pack_mask(a);

  simd<ushort, 16> c = esimd_unpack_mask<16>(b);

  unsigned int d = esimd_pack_mask(c);

  return (b == d);
}

bool test_esimd_min() __attribute__((sycl_device)) {
  simd<short, 16> a(0, 1);
  simd<short, 16> c(0);
  c = esimd_min<short>(a, 5);
  return (c[7] == 5);
}

bool test_esimd_div() __attribute__((sycl_device)) {
  float a = 2.f;
  float b = 1.f;
  float c;
  c = esimd_div_ieee(a, b);
  return (c == 2.f);
}

bool test_esimd_atan() __attribute__((sycl_device)) {
  simd<float, 16> v(0.f, 1.f);
  auto c = esimd_atan(v);
  return (c[0] == 0.f);
}

bool test_esimd_sin_emu() __attribute__((sycl_device)) {
  simd<float, 16> v(0.f, 1.f);
  auto c = esimd_sin_emu(v);
  return (c[0] == 0.f);
}

bool test_esimd_tanh_cody_waite() __attribute__((sycl_device)) {
  simd<float, 16> v(0.f, 1.f);
  auto c = esimd_tanh_cody_waite(v);
  return (c[0] == 0.f);
}

bool test_esimd_dp4() __attribute__((sycl_device)) {
  simd<float, 8> a(0, 1);
  simd<float, 8> b(0, 1);
  simd<float, 8> ret = esimd_dp4(a, b);
  return (ret[0] == ret[1] && ret[1] == ret[2] && ret[2] == ret[3]) &&
         (ret[0] == 14.0f && ret[4] == 126.0f);
}
