// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

// This test checks that esimd_cbit, esimd_fbl and esimd_fbh APIs can be
// compiled by host and device compilers.

#include <sycl/ext/intel/experimental/esimd.hpp>

#include <cstdint>

using namespace sycl::ext::intel::experimental::esimd;

template <typename T, int N> void test_esimd_cbit() SYCL_ESIMD_FUNCTION {
  simd<T, N> v;
  auto cbit_res = esimd_cbit(v);
  auto cbit_scalar_res = esimd_cbit(v[0]);
}

template <typename T, int N> void test_esimd_fbx() SYCL_ESIMD_FUNCTION {
  simd<T, N> v;
  auto fbl_res = esimd_fbl(v);
  auto fbl_scalar_res = esimd_fbl(v[0]);
  auto fbh_res = esimd_fbh(v);
  auto fbh_scalar_res = esimd_fbh(v[0]);
}

void foo() SYCL_ESIMD_FUNCTION {
  test_esimd_cbit<char, 1>();
  test_esimd_cbit<int, 8>();
  test_esimd_cbit<unsigned short, 32>();

  test_esimd_fbx<int, 8>();
  test_esimd_fbx<unsigned int, 16>();
}
