// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s

#include <limits>
#include <sycl/ext/intel/esimd.hpp>
#include <utility>

using namespace sycl::ext::intel::esimd;

// test wrregion size checks.
SYCL_ESIMD_FUNCTION void test_wrregion_size_check() {
  simd<int, 16> v16 = 0;
  v16.template select<64, 1>(0) = slm_block_load<int, 64>(0);
  // expected-error@* {{no matching function for call to '__esimd_wrregion'}}
  // expected-note@sycl/ext/intel/esimd/detail/simd_view_impl.hpp:* {{in instantiation of function template specialization}}
  // expected-note@sycl/ext/intel/esimd/detail/simd_view_impl.hpp:* {{in instantiation of member function}}
  // expected-note@* {{operator=' requested here}}
  // expected-note@sycl/ext/intel/esimd/detail/intrin.hpp:* {{candidate template ignored: requirement '64 <= 16' was not satisfied}}
}
