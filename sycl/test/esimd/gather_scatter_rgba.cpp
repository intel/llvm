// RUN: %clangxx -fsycl -fsyntax-only -Wno-unused-command-line-argument %s

// This test checks compilation of ESIMD slm gather_rgba/scatter_rgba APIs.

#include <CL/sycl.hpp>
#include <limits>
#include <sycl/ext/intel/experimental/esimd.hpp>
#include <utility>

using namespace sycl::ext::intel::experimental::esimd;
using namespace cl::sycl;

void kernel(int *ptr) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, sizeof(int) * 4);
  simd<int, 32 * 4> v1(0, 1);

  auto v0 = gather_rgba<int, 32, rgba_channel_mask::ABGR>(ptr, offsets);

  v0 = v0 + v1;

  scatter_rgba<int, 32, rgba_channel_mask::ABGR>(ptr, offsets, v0);
}
