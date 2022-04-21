// RUN: %clangxx -fsycl -fsyntax-only -Wno-unused-command-line-argument %s

// This test checks compilation of various ESIMD enum types. Those which are
// deprecated must produce deprecation messages.

#include <sycl/ext/intel/experimental/esimd/common.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

void foo() SYCL_ESIMD_FUNCTION {
  int x;

  // These should compile cleanly:
  x = static_cast<int>(split_barrier_action::wait);
  x = static_cast<int>(atomic_op::add);
  x = static_cast<int>(rgba_channel_mask::R);
}
