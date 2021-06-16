// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s

// This test checks compilation of ESIMD slm gather4/scatter4 APIs. Those which
// are deprecated must produce deprecation messages.

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>
#include <limits>
#include <utility>

using namespace sycl::ext::intel::experimental::esimd;
using namespace cl::sycl;

void kernel(accessor<int, 1, access::mode::read_write,
                     access::target::global_buffer> &buf) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);
  simd<int, 32 * 4> v1(0, 1);

  // expected-warning@+2 {{deprecated}}
  // expected-note@sycl/ext/intel/experimental/esimd/common.hpp:* {{}}
  auto v0 = gather4<int, 32, ESIMD_ABGR_ENABLE>(buf.get_pointer(), offsets);
  // expected-warning@+2 {{deprecated}}
  // expected-note@sycl/ext/intel/experimental/esimd/common.hpp:* {{}}
  v0 = gather4<int, 32, ChannelMaskType::ESIMD_ABGR_ENABLE>(buf.get_pointer(),
                                                            offsets);
  v0 = gather4<int, 32, rgba_channel_mask::ABGR>(buf.get_pointer(), offsets);

  v0 = v0 + v1;

  // expected-warning@+2 {{deprecated}}
  // expected-note@sycl/ext/intel/experimental/esimd/common.hpp:* {{}}
  scatter4<int, 32, ESIMD_ABGR_ENABLE>(buf.get_pointer(), v0, offsets);
  // expected-warning@+2 {{deprecated}}
  // expected-note@sycl/ext/intel/experimental/esimd/common.hpp:* {{}}
  scatter4<int, 32, ChannelMaskType::ESIMD_ABGR_ENABLE>(buf.get_pointer(), v0,
                                                        offsets);
  scatter4<int, 32, rgba_channel_mask::ABGR>(buf.get_pointer(), v0, offsets);
}
