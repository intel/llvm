// RUN: %clangxx -fsycl -fsyntax-only -Wno-unused-command-line-argument %s 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

// This test checks compilation of ESIMD atomic APIs. Those which are deprecated
// must produce deprecation messages.

#include <CL/sycl.hpp>
#include <limits>
#include <sycl/ext/intel/experimental/esimd.hpp>
#include <utility>

using namespace sycl::ext::intel::experimental::esimd;
using namespace cl::sycl;

void kernel0(accessor<uint32_t, 1, access::mode::read_write,
                      access::target::global_buffer> &buf) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);

  // CHECK: flat_atomic.cpp:20{{.*}}warning: 'ATOMIC_INC' is deprecated
  // CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
  flat_atomic<EsimdAtomicOpType::ATOMIC_INC, uint32_t, 32>(buf.get_pointer(),
                                                           offsets, 1);
  flat_atomic<atomic_op::inc, uint32_t, 32>(buf.get_pointer(), offsets, 1);
}

void kernel1(accessor<uint32_t, 1, access::mode::read_write,
                      access::target::global_buffer> &buf) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);
  simd<uint32_t, 32> v1(0, 1);

  // CHECK: flat_atomic.cpp:32{{.*}}warning: 'ATOMIC_ADD' is deprecated
  // CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
  flat_atomic<EsimdAtomicOpType::ATOMIC_ADD, uint32_t, 32>(buf.get_pointer(),
                                                           offsets, v1, 1);
  flat_atomic<atomic_op::add, uint32_t, 32>(buf.get_pointer(), offsets, v1, 1);
}

void kernel2(accessor<uint32_t, 1, access::mode::read_write,
                      access::target::global_buffer> &buf) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);
  simd<uint32_t, 32> v1(0, 1);

  // CHECK: flat_atomic.cpp:44{{.*}}warning: 'ATOMIC_CMPXCHG' is deprecated
  // CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
  flat_atomic<EsimdAtomicOpType::ATOMIC_CMPXCHG, uint32_t, 32>(
      buf.get_pointer(), offsets, v1, v1, 1);
  flat_atomic<atomic_op::cmpxchg, uint32_t, 32>(buf.get_pointer(), offsets, v1,
                                                v1, 1);
}

// A "border" between device and host compilations
// CHECK-LABEL: 3 warnings generated
// CHECK: flat_atomic.cpp:20{{.*}}warning: 'ATOMIC_INC' is deprecated
// CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
// CHECK: flat_atomic.cpp:32{{.*}}warning: 'ATOMIC_ADD' is deprecated
// CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
// CHECK: flat_atomic.cpp:44{{.*}}warning: 'ATOMIC_CMPXCHG' is deprecated
// CHECK: sycl/ext/intel/experimental/esimd/common.hpp{{.*}}note:
