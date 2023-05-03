// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s
// Verify simd_view passed to inline asm in l-value context errors, and simd and
// simd_mask work.
#include <sycl/ext/intel/esimd.hpp>

using namespace sycl::ext::intel::esimd;

void test_error() SYCL_ESIMD_FUNCTION {
  simd<float, 16> s;
  simd_mask<16> mask;
  auto view = s.select<8, 1>();
  // expected-error@+1 {{invalid lvalue in asm output}}
  __asm__("%0" : "=rw"(view.data()));

  // expected-error@+1 {{no member named 'data_ref'}}
  __asm__("%0" : "=rw"(view.data_ref()));

  // expected-error@+1 {{invalid lvalue in asm output}}
  __asm__("%0" : "=rw"(s.data()));

  __asm__("%0" : "=rw"(s.data_ref()));

  // expected-error@+1 {{invalid lvalue in asm output}}
  __asm__("%0" : "=rw"(mask.data()));

  __asm__("%0" : "=rw"(mask.data_ref()));
}
