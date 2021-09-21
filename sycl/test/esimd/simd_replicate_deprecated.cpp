// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s

// This test checks deprecated replicate ESIMD APIs

#include <sycl/ext/intel/experimental/esimd.hpp>

using namespace sycl::ext::intel::experimental::esimd;

bool test_replicate1() {
  simd<int, 8> v0(0, 1);
  // expected-warning@+3 2 {{deprecated}}
  // expected-note@sycl/ext/intel/experimental/esimd/simd.hpp:* {{}}
  // expected-note@sycl/ext/intel/experimental/esimd/simd.hpp:* {{has been explicitly marked deprecated here}}
  auto v0_rep = v0.replicate<4, 2>(2);

  return v0[2] == v0_rep[2] && v0[3] == v0_rep[5];
}

bool test_replicate2() {
  simd<int, 8> v0(0, 1);
  // expected-warning@+3 2 {{deprecated}}
  // expected-note@sycl/ext/intel/experimental/esimd/simd.hpp:* {{}}
  // expected-note@sycl/ext/intel/experimental/esimd/simd.hpp:* {{has been explicitly marked deprecated here}}
  auto v0_rep = v0.replicate<2, 4, 2>(1);

  return v0_rep[0] == v0[1] && v0_rep[1] == v0[2] && v0_rep[2] == v0[5];
}

bool test_replicate3() {
  simd<int, 8> v0(0, 1);
  // expected-warning@+3 2 {{deprecated}}
  // expected-note@sycl/ext/intel/experimental/esimd/simd.hpp:* {{}}
  // expected-note@sycl/ext/intel/experimental/esimd/simd.hpp:* {{has been explicitly marked deprecated here}}
  auto v0_rep = v0.replicate<2, 4, 2, 2>(1);

  return v0_rep[0] == v0[1] && v0_rep[1] == v0[3] && v0_rep[2] == v0[5];
}
