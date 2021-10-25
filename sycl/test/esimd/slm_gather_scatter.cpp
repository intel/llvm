// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>
#include <limits>
#include <utility>

using namespace sycl::ext::intel::experimental;
using namespace sycl::ext::intel::experimental::esimd;
using namespace cl::sycl;

void kernel() __attribute__((sycl_device)) {
  simd<uint32_t, 32> offsets(0, sizeof(int));
  simd<int, 32> v1(0, 1);

  auto v0 = slm_gather<int, 32>(offsets);

  fence_mask fm = fence_mask_bit::global_coherent_fence |
                  fence_mask_bit::l3_flush_instructions |
                  fence_mask_bit::l3_flush_texture_data |
                  fence_mask_bit::l3_flush_constant_data |
                  fence_mask_bit::l3_flush_rw_data |
                  fence_mask_bit::local_barrier |
                  fence_mask_bit::l1_flush_ro_data | fence_mask_bit::sw_barrier;

  esimd::fence(fm);
  esimd::barrier();

  v0 = v0 + v1;

  slm_scatter<int, 32>(offsets, v0);
}
