// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>
#include <limits>
#include <utility>

using namespace sycl::INTEL::gpu;
using namespace cl::sycl;

void kernel(accessor<int, 1, access::mode::read_write, access::target::global_buffer> &buf) __attribute__((sycl_device)) {
  simd<int, 32> v1(0, 1);

  auto v0 = block_load<int, 32>(buf.get_pointer());

  v0 = v0 + v1;

  block_store<int, 32>(buf.get_pointer(), v0);
}
