// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s

// This test checks compilation of ESIMD slm load4/store4 APIs. Those which are
// deprecated must produce deprecation messages.

#include <sycl/ext/intel/experimental/esimd.hpp>

using namespace sycl::ext::intel::experimental::esimd;
using namespace cl::sycl;

void caller() SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);
  simd<int, 128> v1(0, 1);

  slm_init(1024);

  // expected-warning@+2 {{deprecated}}
  // expected-note@sycl/ext/intel/experimental/esimd/common.hpp:* {{}}
  auto v0 = slm_load4<int, 32, ESIMD_ABGR_ENABLE>(offsets);
  v0 = slm_load4<int, 32, rgba_channel_mask::ABGR>(offsets);

  v0 = v0 + v1;

  // expected-warning@+2 {{deprecated}}
  // expected-note@sycl/ext/intel/experimental/esimd/common.hpp:* {{}}
  slm_store4<int, 32, ESIMD_ABGR_ENABLE>(v0, offsets);
  slm_store4<int, 32, rgba_channel_mask::ABGR>(v0, offsets);
}
