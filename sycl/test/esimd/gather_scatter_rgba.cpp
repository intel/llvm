// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s

// This test checks that device compiler can:
// - successfully compile gather_rgba/scatter_rgba APIs
// - emit an error if some of the restrictions on template parameters are
//   violated

#include <limits>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <utility>

using namespace sycl::ext::intel::esimd;
using namespace sycl;

void kernel(int *ptr) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, sizeof(int) * 4);
  simd<int, 32 * 4> v1(0, 1);

  auto v0 = gather_rgba<rgba_channel_mask::ABGR>(ptr, offsets);

  v0 = v0 + v1;

  scatter_rgba<rgba_channel_mask::ABGR>(ptr, offsets, v0);
}

constexpr int AGR_N_CHANNELS = 3;

void kernel1(int *ptr, simd<int, 32 * AGR_N_CHANNELS> v) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, sizeof(int) * 4);
  // only 1, 2, 3, 4-element masks covering consequitive channels starting from
  // R are supported
  // expected-error-re@* {{static assertion failed{{.*}}rgba_channel_mask{{.*}}ABGR{{.*}}BGR{{.*}}GR{{.*}}R{{.*}}}}
  // expected-note@* {{in instantiation }}
  // expected-note@+1 {{in instantiation }}
  scatter_rgba<rgba_channel_mask::AGR>(ptr, offsets, v);
}
