// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s

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

void kernel1(accessor<int, 1, access::mode::read_write,
                      access::target::global_buffer> &buf) __SYCL_DEVICE_ATTR {
  simd<int, 32> v1(0, 1);
  // expected-warning@+2 {{deprecated}}
  // expected-note@CL/sycl/INTEL/esimd/esimd_memory.hpp:188 {{}}
  auto v0 = block_load<int, 32>(buf, 0);
  v0 = v0 + v1;
  // expected-warning@+2 {{deprecated}}
  // expected-note@CL/sycl/INTEL/esimd/esimd_memory.hpp:220 {{}}
  block_store<int, 32>(buf, 0, v0);
}

void kernel2(int *ptr) __SYCL_DEVICE_ATTR {
  simd<int, 32> v1(0, 1);
  // expected-warning@+2 {{deprecated}}
  // expected-note@CL/sycl/INTEL/esimd/esimd_memory.hpp:169 {{}}
  auto v0 = block_load<int, 32>(ptr);
  v0 = v0 + v1;
  // expected-warning@+2 {{deprecated}}
  // expected-note@CL/sycl/INTEL/esimd/esimd_memory.hpp:201 {{}}
  block_store<int, 32>(ptr, v0);
}
