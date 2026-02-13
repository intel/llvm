// RUN: not %clangxx -fsycl -fsyntax-only -Wno-unused-command-line-argument %s 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

// This test checks that block_load/store API gets successfully compiled.

#include <limits>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <utility>

using namespace sycl::ext::intel::esimd;
using namespace sycl;

SYCL_EXTERNAL void
kernel1(accessor<int, 1, access::mode::read_write, access::target::device> &buf)
    SYCL_ESIMD_FUNCTION {
  simd<int, 32> v1(0, 1);
  auto v0 = block_load<int, 32>(buf, 0);
  v0 = v0 + v1;
  block_store<int, 32>(buf, 0, v0);
}

SYCL_EXTERNAL void
kernel2(accessor<int, 1, access::mode::read, access::target::device> &buf)
    SYCL_ESIMD_FUNCTION {
  auto v0 = block_load<int, 32>(buf, 0);
}

SYCL_EXTERNAL void
kernel3(accessor<int, 1, access::mode::write, access::target::device> &buf)
    SYCL_ESIMD_FUNCTION {
  simd<int, 32> v1(0, 1);
  block_store<int, 32>(buf, 0, v1);
}

SYCL_EXTERNAL void kernel4(int *ptr) SYCL_ESIMD_FUNCTION {
  simd<int, 32> v1(0, 1);
  auto v0 = block_load<int, 32>(ptr);
  v0 = v0 + v1;
  block_store<int, 32>(ptr, v0);
}

// --- Negative tests.

// Incompatible mode (write).
SYCL_EXTERNAL void
kernel5(accessor<int, 1, access::mode::write, access::target::device> &buf)
    SYCL_ESIMD_FUNCTION {
  simd<int, 32> v;
  // CHECK: block_load_store.cpp:51{{.*}}error: no matching function
  // function for call to 'block_load'
  v = block_load<int, 32>(buf, 0);
}

// Incompatible mode (read).
SYCL_EXTERNAL void
kernel6(accessor<int, 1, access::mode::read, access::target::device> &buf)
    SYCL_ESIMD_FUNCTION {
  simd<int, 32> v(0, 1);
  // CHECK: block_load_store.cpp:61{{.*}}error: no matching function
  // function for call to 'block_store'
  block_store<int, 32>(buf, 0, v);
}

// Incompatible mode (read).
SYCL_EXTERNAL void
kernel7(local_accessor<const int, 1> &buf) SYCL_ESIMD_FUNCTION {
  simd<int, 32> v(0, 1);
  // CHECK: block_load_store.cpp:70{{.*}}error: no matching function
  // function for call to 'block_store'
  block_store<int, 32>(buf, 0, v);
}

// - Positive cases
SYCL_EXTERNAL void
kernel7(accessor<int, 1, access::mode::read_write, access::target::device> &buf)
    SYCL_ESIMD_FUNCTION {
  simd<int, 32> v1(0, 1);
  auto vals_view = v1.select<8, 1>(0);
  block_store<int, 8>(buf, 0, vals_view);
}

SYCL_EXTERNAL void kernel8(local_accessor<int, 1> &buf) SYCL_ESIMD_FUNCTION {
  simd<int, 32> v1(0, 1);
  auto vals_view = v1.select<8, 1>(0);
  block_store<int, 8>(buf, 0, vals_view);
}

SYCL_EXTERNAL void kernel9(int *ptr) SYCL_ESIMD_FUNCTION {
  simd<int, 32> v1(0, 1);
  auto vals_view = v1.select<8, 1>(0);
  block_store<int, 8>(ptr, vals_view);
}