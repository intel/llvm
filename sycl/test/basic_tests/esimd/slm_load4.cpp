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
  simd<int, 128> v1(0, 1);

  slm_init(1024);

  auto v0 = slm_load4<int, 32, ESIMD_ABGR_ENABLE>(offsets);

  v0 = v0 + v1;

  slm_store4<int, 32, ESIMD_ABGR_ENABLE>(v0, offsets);
}
