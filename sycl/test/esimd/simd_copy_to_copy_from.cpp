// RUN: not %clangxx -fsycl -fsycl-device-only -fsyntax-only -Wno-unused-command-line-argument %s 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"
// RUN: not %clangxx %fsycl-host-only -fsyntax-only -Wno-unused-command-line-argument %s 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

// This test checks that both host and device compilers can:
// - successfully compile simd::copy_to and simd::copy_from APIs
// - emit an error if argument of an incompatible type is used
//   in place of the accessor argument

#include <CL/sycl.hpp>
#include <limits>
#include <sycl/ext/intel/experimental/esimd.hpp>
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
  // CHECK: simd_copy_to_copy_from.cpp:46{{.*}}error: no matching member function for call to 'copy_from'
  v0.copy_from(buf, 0);
  v0 = v0 + v1;
  // CHECK: simd_copy_to_copy_from.cpp:49{{.*}}error: no matching member function for call to 'copy_to'
  v0.copy_to(buf, 0);
}

// Incompatible mode (write).
SYCL_EXTERNAL void kernel4(
    accessor<int, 1, access::mode::write, access::target::global_buffer> &buf)
    SYCL_ESIMD_FUNCTION {
  simd<int, 32> v;
  // CHECK: simd_copy_to_copy_from.cpp:58{{.*}}error: no matching member function for call to 'copy_from'
  v.copy_from(buf, 0);
}

// Incompatible mode (read).
SYCL_EXTERNAL void kernel5(
    accessor<int, 1, access::mode::read, access::target::global_buffer> &buf)
    SYCL_ESIMD_FUNCTION {
  simd<int, 32> v(0, 1);
  // CHECK: simd_copy_to_copy_from.cpp:67{{.*}}error: no matching member function for call to 'copy_to'
  v.copy_to(buf, 0);
}
