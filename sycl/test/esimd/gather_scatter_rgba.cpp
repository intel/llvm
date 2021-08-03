// RUN: %clangxx -fsycl -fsyntax-only -Wno-unused-command-line-argument %s 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

// This test checks compilation of ESIMD slm gather_rgba/scatter_rgba APIs.
// Those which are deprecated must produce deprecation messages.

#include <CL/sycl.hpp>
#include <limits>
#include <sycl/ext/intel/experimental/esimd.hpp>
#include <utility>

using namespace sycl::ext::intel::experimental::esimd;
using namespace cl::sycl;

void kernel(accessor<int, 1, access::mode::read_write,
                     access::target::global_buffer> &buf) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);
  simd<int, 32 * 4> v1(0, 1);

  // CHECK: gather_scatter_rgba.cpp:21{{.*}}warning: 'ESIMD_ABGR_ENABLE' is deprecated
  // CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
  auto v0 = gather_rgba<int, 32, ESIMD_ABGR_ENABLE>(buf.get_pointer(), offsets);
  // CHECK: gather_scatter_rgba.cpp:24{{.*}}warning: 'ESIMD_ABGR_ENABLE' is deprecated
  // CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
  v0 = gather_rgba<int, 32, ChannelMaskType::ESIMD_ABGR_ENABLE>(
      buf.get_pointer(), offsets);
  v0 =
      gather_rgba<int, 32, rgba_channel_mask::ABGR>(buf.get_pointer(), offsets);

  v0 = v0 + v1;

  // CHECK: gather_scatter_rgba.cpp:33{{.*}}warning: 'ESIMD_ABGR_ENABLE' is deprecated
  // CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
  scatter_rgba<int, 32, ESIMD_ABGR_ENABLE>(buf.get_pointer(), v0, offsets);
  // CHECK: gather_scatter_rgba.cpp:36{{.*}}warning: 'ESIMD_ABGR_ENABLE' is deprecated
  // CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
  scatter_rgba<int, 32, ChannelMaskType::ESIMD_ABGR_ENABLE>(buf.get_pointer(),
                                                            v0, offsets);
  scatter_rgba<int, 32, rgba_channel_mask::ABGR>(buf.get_pointer(), v0,
                                                 offsets);
}

// A "border" between host and device compilations
// CHECK-LABEL: 4 warnings generated
// CHECK: gather_scatter_rgba.cpp:21{{.*}}warning: 'ESIMD_ABGR_ENABLE' is deprecated
// CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
// CHECK: gather_scatter_rgba.cpp:24{{.*}}warning: 'ESIMD_ABGR_ENABLE' is deprecated
// CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
// CHECK: gather_scatter_rgba.cpp:33{{.*}}warning: 'ESIMD_ABGR_ENABLE' is deprecated
// CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
// CHECK: gather_scatter_rgba.cpp:36{{.*}}warning: 'ESIMD_ABGR_ENABLE' is deprecated
// CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
