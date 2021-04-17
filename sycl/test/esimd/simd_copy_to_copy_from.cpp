// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s

// This test checks that both host and device compilers can:
// - successfully compile simd::copy_to and simd::copy_from APIs
// - emit an error if argument of an incompatible type is used
//   in place of the accessor argument

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>
#include <limits>
#include <utility>

using namespace sycl::INTEL::gpu;
using namespace cl::sycl;

#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_DEVICE_ATTR __attribute__((sycl_device))
#else
#define __SYCL_DEVICE_ATTR
#endif // __SYCL_DEVICE_ONLY__

// --- Postive tests.

void kernel1(accessor<int, 1, access::mode::read_write,
                      access::target::global_buffer> &buf) __SYCL_DEVICE_ATTR {
  simd<int, 32> v1(0, 1);
  simd<int, 32> v0;
  v0.copy_from(buf, 0);
  v0 = v0 + v1;
  v0.copy_to(buf, 0);
}

void kernel2(int *ptr) __SYCL_DEVICE_ATTR {
  simd<int, 32> v1(0, 1);
  simd<int, 32> v0;
  v0.copy_from(ptr);
  v0 = v0 + v1;
  v0.copy_to(ptr);
}

// --- Negative tests.

// Incompatible target.
void kernel3(accessor<int, 1, access::mode::read_write, access::target::local>
                 &buf) __SYCL_DEVICE_ATTR {
  simd<int, 32> v1(0, 1);
  simd<int, 32> v0;
  // expected-error@+3 {{no matching member function for call to 'copy_from'}}
  // expected-note@CL/sycl/INTEL/esimd/esimd.hpp:513 {{}}
  // expected-note@CL/sycl/INTEL/esimd/esimd.hpp:508 {{}}
  v0.copy_from(buf, 0);
  v0 = v0 + v1;
  // expected-error@+3 {{no matching member function for call to 'copy_to'}}
  // expected-note@CL/sycl/INTEL/esimd/esimd.hpp:497 {{}}
  // expected-note@CL/sycl/INTEL/esimd/esimd.hpp:523 {{}}
  v0.copy_to(buf, 0);
}

// Incompatible mode (write).
void kernel4(
    accessor<int, 1, access::mode::write, access::target::global_buffer> &buf)
    __SYCL_DEVICE_ATTR {
  simd<int, 32> v;
  // expected-error@+3 {{no matching member function for call to 'copy_from'}}
  // expected-note@CL/sycl/INTEL/esimd/esimd.hpp:513 {{}}
  // expected-note@CL/sycl/INTEL/esimd/esimd.hpp:508 {{}}
  v.copy_from(buf, 0);
}

// Incompatible mode (read).
void kernel5(accessor<int, 1, access::mode::read, access::target::global_buffer>
                 &buf) __SYCL_DEVICE_ATTR {
  simd<int, 32> v(0, 1);
  // expected-error@+3 {{no matching member function for call to 'copy_to'}}
  // expected-note@CL/sycl/INTEL/esimd/esimd.hpp:497 {{}}
  // expected-note@CL/sycl/INTEL/esimd/esimd.hpp:523 {{}}
  v.copy_to(buf, 0);
}
