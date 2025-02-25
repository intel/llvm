// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <limits>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <utility>

using namespace sycl::ext::intel;
using namespace sycl::ext::intel::esimd;
using namespace sycl;

void kernel() __attribute__((sycl_device)) {
  simd<uint32_t, 32> offsets(0, sizeof(int));
  simd<int, 32> v1(0, 1);

  auto v0 = slm_gather<int, 32>(offsets);

  constexpr auto fm =
      fence_mask::global_coherent_fence | fence_mask::l2_flush_instructions |
      fence_mask::l2_flush_texture_data | fence_mask::l2_flush_constant_data |
      fence_mask::l2_flush_rw_data | fence_mask::local_barrier |
      fence_mask::l1_flush_ro_data;

  esimd::fence<fm>();
  esimd::barrier();

  v0 = v0 + v1;

  slm_scatter<int, 32>(offsets, v0);
}
