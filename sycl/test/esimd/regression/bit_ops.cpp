// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

// This test checks that esimd::cbit, esimd::fbl and esimd::fbh APIs can be
// compiled by host and device compilers.

#include <sycl/ext/intel/esimd.hpp>

#include <cstdint>

using namespace sycl::ext::intel;
using namespace sycl::ext::intel::esimd;

template <typename T, int N> void test_esimd_cbit() SYCL_ESIMD_FUNCTION {
  simd<T, N> v;
  auto cbit_res = esimd::cbit(v);
  auto cbit_scalar_res = esimd::cbit(v[0]);
}

template <typename T, int N> void test_esimd_fbx() SYCL_ESIMD_FUNCTION {
  simd<T, N> v;
  auto fbl_res = esimd::fbl(v);
  auto fbl_scalar_res = esimd::fbl(v[0]);
  auto fbh_res = esimd::fbh(v);
  auto fbh_scalar_res = esimd::fbh(v[0]);
}

void foo() SYCL_ESIMD_FUNCTION {
  test_esimd_cbit<char, 1>();
  test_esimd_cbit<int, 8>();
  test_esimd_cbit<unsigned short, 32>();

  test_esimd_fbx<int, 8>();
  test_esimd_fbx<unsigned int, 16>();
}
