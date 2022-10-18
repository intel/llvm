// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <limits>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <utility>

using namespace sycl::ext::intel::esimd;
using namespace sycl;

void kernel() __attribute__((sycl_device)) {
  simd<int, 32> v1(0, 1);

  auto v0 = slm_block_load<int, 32>(0);

  v0 = v0 + v1;

  slm_block_store<int, 32>(0, v0);
}
