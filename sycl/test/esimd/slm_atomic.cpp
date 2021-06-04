// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify %s

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

  // expected-warning@+2 {{deprecated}}
  // expected-note@sycl/ext/intel/experimental/esimd/common.hpp:* {{}}
  slm_atomic<EsimdAtomicOpType::ATOMIC_INC, uint32_t, 32>(offsets, 1);
  slm_atomic<atomic_op::inc, uint32_t, 32>(offsets, 1);
}

void kernel1() SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);
  simd<uint32_t, 32> v1(0, 1);

  // expected-warning@+2 {{deprecated}}
  // expected-note@sycl/ext/intel/experimental/esimd/common.hpp:* {{}}
  slm_atomic<EsimdAtomicOpType::ATOMIC_ADD, uint32_t, 32>(offsets, v1, 1);
  slm_atomic<atomic_op::add, uint32_t, 32>(offsets, v1, 1);
}

void kernel2() SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);
  simd<uint32_t, 32> v1(0, 1);

  // expected-warning@+2 {{deprecated}}
  // expected-note@sycl/ext/intel/experimental/esimd/common.hpp:* {{}}
  slm_atomic<EsimdAtomicOpType::ATOMIC_CMPXCHG, uint32_t, 32>(offsets, v1, v1, 1);
  slm_atomic<atomic_op::cmpxchg, uint32_t, 32>(offsets, v1, v1, 1);
}
