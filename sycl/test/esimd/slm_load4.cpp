// RUN: %clangxx -fsycl -fsyntax-only -Wno-unused-command-line-argument %s 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

// This test checks compilation of ESIMD slm load4/store4 APIs. Those which are
// deprecated must produce deprecation messages.

#include <sycl/ext/intel/experimental/esimd.hpp>

using namespace sycl::ext::intel::experimental::esimd;
using namespace cl::sycl;

void caller() SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);
  simd<int, 128> v1(0, 1);

  slm_init(1024);

  // CHECK: slm_load4.cpp:19{{.*}}warning: 'ESIMD_ABGR_ENABLE' is deprecated
  // CHECK: sycl/ext/intel/experimental/esimd/common.hpp:{{.*}}note:
  auto v0 = slm_load4<int, 32, ESIMD_ABGR_ENABLE>(offsets);
  v0 = slm_load4<int, 32, rgba_channel_mask::ABGR>(offsets);

  v0 = v0 + v1;

  // CHECK: slm_load4.cpp:26{{.*}}warning: 'ESIMD_ABGR_ENABLE' is deprecated
  // CHECK: sycl/ext/intel/experimental/esimd/common.hpp:{{.*}}note:
  slm_store4<int, 32, ESIMD_ABGR_ENABLE>(v0, offsets);
  slm_store4<int, 32, rgba_channel_mask::ABGR>(v0, offsets);
}

// A "border" between host and device compilations
// CHECK-LABEL: 2 warnings generated
// CHECK: slm_load4.cpp:19{{.*}}warning: 'ESIMD_ABGR_ENABLE' is deprecated
// CHECK: sycl/ext/intel/experimental/esimd/common.hpp:{{.*}}note:
// CHECK: slm_load4.cpp:26{{.*}}warning: 'ESIMD_ABGR_ENABLE' is deprecated
// CHECK: sycl/ext/intel/experimental/esimd/common.hpp:{{.*}}note:
