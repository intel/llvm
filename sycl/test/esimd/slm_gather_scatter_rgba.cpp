// RUN: %clangxx -fsycl -fsyntax-only -Wno-unused-command-line-argument %s

// This test checks compilation of ESIMD slm_gather_rgba/slm_scatter_rgba APIs.

#include <sycl/ext/intel/experimental/esimd.hpp>

using namespace sycl::ext::intel::experimental::esimd;
using namespace cl::sycl;

void caller() SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, sizeof(int) * 4);
  simd<int, 128> v1(0, 1);

  slm_init(1024);

  auto v0 = slm_gather_rgba<int, 32, rgba_channel_mask::ABGR>(offsets);

  v0 = v0 + v1;

  slm_scatter_rgba<int, 32, rgba_channel_mask::ABGR>(offsets, v0);
}
