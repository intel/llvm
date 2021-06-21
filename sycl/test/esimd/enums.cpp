// RUN: %clangxx -fsycl -fsyntax-only %s 2>&1 | FileCheck %s

// This test checks compilation of various ESIMD enum types. Those which are
// deprecated must produce deprecation messages.

#include <sycl/ext/intel/experimental/esimd/common.hpp>

using namespace sycl::ext::intel::experimental::esimd;

void foo() SYCL_ESIMD_FUNCTION {
  // These should produce deprecation messages for both device:
  int x;
  // CHECK: enums.cpp:15{{.*}}warning: 'WAIT' is deprecated
  // CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
  x = static_cast<int>(ESIMD_SBARRIER_WAIT);
  // CHECK: enums.cpp:18{{.*}}warning: 'ATOMIC_ADD' is deprecated
  // CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
  x = static_cast<int>(EsimdAtomicOpType::ATOMIC_ADD);
  // CHECK: enums.cpp:21{{.*}}warning: 'ESIMD_R_ENABLE' is deprecated
  // CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
  x = static_cast<int>(ChannelMaskType::ESIMD_R_ENABLE);
  // CHECK: enums.cpp:24{{.*}}warning: 'GENX_NOSAT' is deprecated
  // CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
  x = static_cast<int>(GENX_NOSAT);

  // A "border" between host and device compilations
  // CHECK-LABEL: 4 warnings generated

  // And for host:
  // CHECK: enums.cpp:15{{.*}}warning: 'WAIT' is deprecated
  // CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
  // CHECK: enums.cpp:18{{.*}}warning: 'ATOMIC_ADD' is deprecated
  // CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
  // CHECK: enums.cpp:21{{.*}}warning: 'ESIMD_R_ENABLE' is deprecated
  // CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
  // CHECK: enums.cpp:24{{.*}}warning: 'GENX_NOSAT' is deprecated
  // CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:

  // These should compile cleanly:
  x = static_cast<int>(split_barrier_action::wait);
  x = static_cast<int>(atomic_op::add);
  x = static_cast<int>(rgba_channel_mask::R);
  x = static_cast<int>(saturation::off);
}
