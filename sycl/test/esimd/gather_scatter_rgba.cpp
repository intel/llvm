// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s

// This test checks that device compiler can:
// - successfully compile gather_rgba/scatter_rgba APIs
// - emit an error if some of the restrictions on template parameters are
//   violated

#include <CL/sycl.hpp>
#include <limits>
#include <sycl/ext/intel/esimd.hpp>
#include <utility>

using namespace sycl::ext::intel::esimd;
using namespace cl::sycl;

void kernel(int *ptr) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, sizeof(int) * 4);
  simd<int, 32 * 4> v1(0, 1);

  auto v0 = gather_rgba<int, 32, rgba_channel_mask::ABGR>(ptr, offsets);

  v0 = v0 + v1;

  scatter_rgba<int, 32, rgba_channel_mask::ABGR>(ptr, offsets, v0);
}

constexpr int AGR_N_CHANNELS = 3;

void kernel1(int *ptr, simd<int, 32 * AGR_N_CHANNELS> v) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, sizeof(int) * 4);
  // only 1, 2, 3, 4-element masks covering consequitive channels starting from
  // R are supported
  // expected-error-re@* {{static_assert failed{{.*}}Only ABGR, BGR, GR, R channel masks are valid in write operations}}
  // expected-note@* {{in instantiation }}
  // expected-note@+1 {{in instantiation }}
  scatter_rgba<int, 32, rgba_channel_mask::AGR>(ptr, offsets, v);
}
