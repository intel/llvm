// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s

#include <limits>
#include <sycl/ext/intel/esimd.hpp>
#include <utility>

using namespace sycl::ext::intel::esimd;

// test wrregion size checks.
SYCL_ESIMD_FUNCTION void test_wrregion_size_check() {
  simd<int, 16> v16 = 0;
  simd<int, 64> v64;
  v16.template select<64, 1>(0) = v64;
  // expected-error@sycl/ext/intel/esimd/detail/util.hpp:* {{static assertion failed due to requirement '64 <= 16'}}
  // expected-note@sycl/ext/intel/esimd/detail/simd_obj_impl.hpp:* {{in instantiation of function template specialization}}
  // expected-note@sycl/ext/intel/esimd/detail/simd_view_impl.hpp:* {{in instantiation of function template specialization}}
  // expected-note@sycl/ext/intel/esimd/detail/simd_view_impl.hpp:* {{in instantiation of member function}}
  // expected-note@* {{in instantiation of member function}}

  // expected-error@sycl/ext/intel/esimd/detail/util.hpp:* {{static assertion failed due to requirement '(64 - 1) * 1 < 16'}}
  // expected-note@* {{expression evaluates to '63 < 16'}}

  // expected-error@* {{no matching function for call to '__esimd_wrregion'}}
  // expected-note@sycl/ext/intel/esimd/detail/simd_view_impl.hpp:* {{in instantiation of function template specialization}}
  // expected-note@sycl/ext/intel/esimd/detail/simd_view_impl.hpp:* {{in instantiation of member function}}
  // expected-note@* {{in instantiation of member function}}
  // expected-note@sycl/ext/intel/esimd/detail/intrin.hpp:* {{candidate template ignored: requirement '64 <= 16' was not satisfied}}

  simd<int, 2> v2;
  v16.template select<2, 64>() = v2;
  // expected-error@sycl/ext/intel/esimd/detail/util.hpp:* {{static assertion failed due to requirement '(2 - 1) * 64 < 16'}}
  // expected-note@sycl/ext/intel/esimd/detail/simd_obj_impl.hpp:* {{in instantiation of function template specialization}}
  // expected-note@sycl/ext/intel/esimd/detail/simd_view_impl.hpp:* {{in instantiation of function template specialization}}
  // expected-note@sycl/ext/intel/esimd/detail/simd_view_impl.hpp:* {{in instantiation of member function}}
  // expected-note@* {{in instantiation of member function}}
  // expected-note@* {{expression evaluates to '64 < 16'}}
}
