// RUN: %clangxx -fsycl -fsycl-explicit-simd -fsycl-device-only -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <CL/sycl.hpp>
#include <CL/sycl/intel/esimd.hpp>
#include <limits>
#include <utility>

using namespace sycl::intel::gpu;
using namespace cl::sycl;

void kernel() __attribute__((sycl_device)) {
  simd<uint32_t, 32> offsets(0, 1);
  simd<int, 32> v1(0, 1);

  auto v0 = slm_load<int, 32>(offsets);

  slm_fence(3);
  esimd_barrier();

  v0 = v0 + v1;

  slm_store<int, 32>(v0, offsets);
}
