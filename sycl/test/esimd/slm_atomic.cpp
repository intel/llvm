// RUN: %clangxx -fsycl -fsyntax-only -Wno-unused-command-line-argument %s 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

// This test checks compilation of ESIMD slm atomic APIs. Those which are
// deprecated must produce deprecation messages.

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>
#include <limits>
#include <utility>

using namespace sycl::ext::intel::experimental::esimd;
using namespace cl::sycl;

void kernel0() SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);

  // CHECK: slm_atomic.cpp:19{{.*}}warning: 'ATOMIC_INC' is deprecated
  // CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
  slm_atomic<EsimdAtomicOpType::ATOMIC_INC, uint32_t, 32>(offsets, 1);
  slm_atomic<atomic_op::inc, uint32_t, 32>(offsets, 1);
}

void kernel1() SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);
  simd<uint32_t, 32> v1(0, 1);

  // CHECK: slm_atomic.cpp:29{{.*}}warning: 'ATOMIC_ADD' is deprecated
  // CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
  slm_atomic<EsimdAtomicOpType::ATOMIC_ADD, uint32_t, 32>(offsets, v1, 1);
  slm_atomic<atomic_op::add, uint32_t, 32>(offsets, v1, 1);
}

void kernel2() SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);
  simd<uint32_t, 32> v1(0, 1);

  // CHECK: slm_atomic.cpp:39{{.*}}warning: 'ATOMIC_CMPXCHG' is deprecated
  // CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
  slm_atomic<EsimdAtomicOpType::ATOMIC_CMPXCHG, uint32_t, 32>(offsets, v1, v1, 1);
  slm_atomic<atomic_op::cmpxchg, uint32_t, 32>(offsets, v1, v1, 1);
}

// A "border" between device and host compilations
// CHECK-LABEL: 3 warnings generated
// CHECK: slm_atomic.cpp:19{{.*}}warning: 'ATOMIC_INC' is deprecated
// CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
// CHECK: slm_atomic.cpp:29{{.*}}warning: 'ATOMIC_ADD' is deprecated
// CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
// CHECK: slm_atomic.cpp:39{{.*}}warning: 'ATOMIC_CMPXCHG' is deprecated
// CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
