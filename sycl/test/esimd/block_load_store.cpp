// RUN: %clangxx -fsycl -fsyntax-only %s

// This test checks that block_load/store API gets successfully compiled.

#include <CL/sycl.hpp>
#include <limits>
#include <sycl/ext/intel/experimental/esimd.hpp>
#include <utility>

using namespace sycl::ext::intel::experimental::esimd;
using namespace cl::sycl;

SYCL_EXTERNAL void kernel1(
    accessor<int, 1, access::mode::read_write, access::target::global_buffer>
        &buf) SYCL_ESIMD_FUNCTION {
  simd<int, 32> v1(0, 1);
  auto v0 = block_load<int, 32>(buf, 0);
  v0 = v0 + v1;
  block_store<int, 32>(buf, 0, v0);
}

SYCL_EXTERNAL void kernel2(int *ptr) SYCL_ESIMD_FUNCTION {
  simd<int, 32> v1(0, 1);
  auto v0 = block_load<int, 32>(ptr);
  v0 = v0 + v1;
  block_store<int, 32>(ptr, v0);
}
