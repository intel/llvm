// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s

// This test checks that both host and device compilers can:
// - successfully compile simd::copy_to and simd::copy_from APIs
// - emit an error if argument of an incompatible type is used
//   in place of the accessor argument

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>
#include <limits>
#include <utility>

using namespace sycl::ext::intel::experimental::esimd;
using namespace cl::sycl;

// --- Postive tests.

SYCL_EXTERNAL void kernel1(
    accessor<int, 1, access::mode::read_write, access::target::global_buffer>
        &buf) SYCL_ESIMD_FUNCTION {
  simd<int, 32> v1(0, 1);
  simd<int, 32> v0;
  v0.copy_from(buf, 0);
  v0 = v0 + v1;
  v0.copy_to(buf, 0);
}

SYCL_EXTERNAL void kernel2(int *ptr) SYCL_ESIMD_FUNCTION {
  simd<int, 32> v1(0, 1);
  simd<int, 32> v0;
  v0.copy_from(ptr);
  v0 = v0 + v1;
  v0.copy_to(ptr);
}

// --- Negative tests.

// Incompatible target.
SYCL_EXTERNAL void
kernel3(accessor<int, 1, access::mode::read_write, access::target::local> &buf)
    SYCL_ESIMD_FUNCTION {
  simd<int, 32> v1(0, 1);
  simd<int, 32> v0;
  // expected-error@+3 {{no matching member function for call to 'copy_from'}}
  // expected-note@sycl/ext/intel/experimental/esimd/simd.hpp:* {{}}
  // expected-note@sycl/ext/intel/experimental/esimd/simd.hpp:* {{}}
  v0.copy_from(buf, 0);
  v0 = v0 + v1;
  // expected-error@+3 {{no matching member function for call to 'copy_to'}}
  // expected-note@sycl/ext/intel/experimental/esimd/simd.hpp:* {{}}
  // expected-note@sycl/ext/intel/experimental/esimd/simd.hpp:* {{}}
  v0.copy_to(buf, 0);
}

// Incompatible mode (write).
SYCL_EXTERNAL void kernel4(
    accessor<int, 1, access::mode::write, access::target::global_buffer> &buf)
    SYCL_ESIMD_FUNCTION {
  simd<int, 32> v;
  // expected-error@+3 {{no matching member function for call to 'copy_from'}}
  // expected-note@sycl/ext/intel/experimental/esimd/simd.hpp:* {{}}
  // expected-note@sycl/ext/intel/experimental/esimd/simd.hpp:* {{}}
  v.copy_from(buf, 0);
}

// Incompatible mode (read).
SYCL_EXTERNAL void kernel5(
    accessor<int, 1, access::mode::read, access::target::global_buffer> &buf)
    SYCL_ESIMD_FUNCTION {
  simd<int, 32> v(0, 1);
  // expected-error@+3 {{no matching member function for call to 'copy_to'}}
  // expected-note@sycl/ext/intel/experimental/esimd/simd.hpp:* {{}}
  // expected-note@sycl/ext/intel/experimental/esimd/simd.hpp:* {{}}
  v.copy_to(buf, 0);
}
