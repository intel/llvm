// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s

// This test checks compilation of various ESIMD enum types. Those which are
// deprecated must produce deprecation messages.

#include <sycl/ext/intel/experimental/esimd/common.hpp>

using namespace sycl::ext::intel::experimental::esimd;

void foo() SYCL_ESIMD_FUNCTION {
  // These should produce deprecation messages:
  int x;
  // expected-warning@+2 {{deprecated}}
  // expected-note@sycl/ext/intel/experimental/esimd/common.hpp:* {{}}
  x = static_cast<int>(ESIMD_SBARRIER_WAIT);
  // expected-warning@+2 {{deprecated}}
  // expected-note@sycl/ext/intel/experimental/esimd/common.hpp:* {{}}
  x = static_cast<int>(EsimdAtomicOpType::ATOMIC_ADD);
  // expected-warning@+2 {{deprecated}}
  // expected-note@sycl/ext/intel/experimental/esimd/common.hpp:* {{}}
  x = static_cast<int>(ChannelMaskType::ESIMD_R_ENABLE);
  // expected-warning@+2 {{deprecated}}
  // expected-note@sycl/ext/intel/experimental/esimd/common.hpp:* {{}}
  x = static_cast<int>(GENX_NOSAT);

  // These should compile cleanly:
  x = static_cast<int>(split_barrier_action::wait);
  x = static_cast<int>(atomic_op::add);
  x = static_cast<int>(rgba_channel_mask::R);
  x = static_cast<int>(saturation::off);
}
