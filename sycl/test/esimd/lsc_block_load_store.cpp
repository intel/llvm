// RUN: not %clangxx -fsycl -fsyntax-only -Wno-unused-command-line-argument %s 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

// This test checks that lsc_block_load/store API gets successfully compiled.

#include <limits>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <utility>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;
using namespace sycl;

SYCL_EXTERNAL void
kernel1(accessor<int, 1, access::mode::read_write, access::target::device> &buf)
    SYCL_ESIMD_FUNCTION {
  simd<int, 32> v1(0, 1);
  auto v0 = lsc_block_load<int, 32>(buf, 0);
  v0 = v0 + v1;
  lsc_block_store<int, 32>(buf, 0, v0);
}

SYCL_EXTERNAL void kernel2(int *ptr) SYCL_ESIMD_FUNCTION {
  simd<int, 32> v1(0, 1);
  auto v0 = lsc_block_load<int, 32>(ptr);
  v0 = v0 + v1;
  lsc_block_store<int, 32>(ptr, v0);
}

// --- Negative tests.

// Incompatible mode (write).
SYCL_EXTERNAL void
kernel4(accessor<int, 1, access::mode::write, access::target::device> &buf)
    SYCL_ESIMD_FUNCTION {
  simd<int, 32> v;
  // CHECK: lsc_block_load_store.cpp:39{{.*}}error: no matching function
  // function for call to 'lsc_block_load'
  v = lsc_block_load<int, 32>(buf, 0);
}

// Incompatible mode (read).
SYCL_EXTERNAL void
kernel5(accessor<int, 1, access::mode::read, access::target::device> &buf)
    SYCL_ESIMD_FUNCTION {
  simd<int, 32> v(0, 1);
  // CHECK: lsc_block_load_store.cpp:49{{.*}}error: no matching function
  // function for call to 'lsc_block_store'
  lsc_block_store<int, 32>(buf, 0, v);
}

// Incompatible mode (read).
SYCL_EXTERNAL void
kernel6(local_accessor<const int, 1> &buf) SYCL_ESIMD_FUNCTION {
  simd<int, 32> v(0, 1);
  // CHECK: lsc_block_load_store.cpp:58{{.*}}error: no matching function
  // function for call to 'lsc_block_store'
  lsc_block_store<int, 32>(buf, 0, v);
}
