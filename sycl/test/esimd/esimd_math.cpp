// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <limits>
#include <sycl/ext/intel/esimd.hpp>
#include <utility>

using namespace sycl::ext::intel;
using namespace sycl::ext::intel::esimd;

bool test_esimd_mask() __attribute__((sycl_device)) {
  simd_mask<16> a(0);
  a.select<4, 1>(4) = 1;
  a.select<4, 1>(12) = 1;
  unsigned int b = esimd::pack_mask(a);

  simd_mask<16> c = esimd::unpack_mask<16>(b);

  unsigned int d = esimd::pack_mask(c);

  return (b == d);
}

bool test_esimd_min() __attribute__((sycl_device)) {
  simd<short, 16> a(0, 1);
  simd<short, 16> c(0);
  c = esimd::min<short>(a, 5);
  return (c[7] == 5);
}

bool test_esimd_div() __attribute__((sycl_device)) {
  float a = 2.f;
  float b = 1.f;
  float c;
  c = esimd::div_ieee(a, b);
  return (c == 2.f);
}

bool test_esimd_atan() __attribute__((sycl_device)) {
  simd<float, 16> v(0.f, 1.f);
  auto c = experimental::esimd::atan(v);
  return (c[0] == 0.f);
}

bool test_esimd_sin_emu() __attribute__((sycl_device)) {
  simd<float, 16> v(0.f, 1.f);
  auto c = experimental::esimd::sin_emu(v);
  return (c[0] == 0.f);
}

bool test_esimd_tanh_cody_waite() __attribute__((sycl_device)) {
  simd<float, 16> v(0.f, 1.f);
  auto c = experimental::esimd::tanh_cody_waite(v);
  return (c[0] == 0.f);
}

bool test_esimd_dp4() __attribute__((sycl_device)) {
  simd<float, 8> a(0, 1);
  simd<float, 8> b(0, 1);
  simd<float, 8> ret = experimental::esimd::dp4(a, b);
  return (ret[0] == ret[1] && ret[1] == ret[2] && ret[2] == ret[3]) &&
         (ret[0] == 14.0f && ret[4] == 126.0f);
}

bool test_esimd_trunc() __attribute__((sycl_device)) {
  simd<float, 16> vfa(1.4f);
  simd<float, 16> vfr = experimental::esimd::trunc<float, 16>(vfa);
  simd<short, 16> vsr = experimental::esimd::trunc<short, 16>(vfa);

  float sfa = 2.8f;
  float sfr = experimental::esimd::trunc<float>(sfa);
  short ssr = experimental::esimd::trunc<short>(sfa);

  return (vfr[0] == 1.f) && (vsr[0] == 1) && (sfr == 2.f) && (ssr == 2);
}

bool test_esimd_ballot() __attribute__((sycl_device)) {
  simd<ushort, 4> vus4({1, 0, 3, 0});
  simd<ushort, 8> vus8({1, 0, 3, 0, 5, 0, 7, 0});
  simd<ushort, 20> vus20(
      {1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 1, 0, 3, 0, 5, 0, 7, 0, 9, 0});

  uint mus4 = esimd::ballot(vus4);
  uint mus8 = esimd::ballot(vus8);
  uint mus20 = esimd::ballot(vus20);

  simd<uint, 4> vui4({1, 0, 3, 0});
  simd<uint, 16> vui16({1, 0, 3, 0, 5, 0, 7, 0, 1, 0, 3, 0, 5, 0, 7, 0});
  simd<uint, 20> vui20(
      {1, 0, 3, 0, 5, 0, 7, 0, 9, 0, 1, 0, 3, 0, 5, 0, 7, 0, 9, 0});

  uint mui4 = esimd::ballot(vui4);
  uint mui16 = esimd::ballot(vui16);
  uint mui20 = esimd::ballot(vui20);

  return (mus4 == 0x5) && (mus8 == 0x55) && (mus20 = 0x55555) &&
         (mui4 == 0x5) && (mui16 == 0x5555) && (mui20 = 0x55555);
}
