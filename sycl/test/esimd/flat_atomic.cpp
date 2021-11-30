// RUN: %clangxx -fsycl -fsyntax-only -Wno-unused-command-line-argument %s

// This test checks compilation of ESIMD atomic APIs.

#include <CL/sycl.hpp>
#include <limits>
#include <sycl/ext/intel/experimental/esimd.hpp>
#include <utility>

using namespace sycl::ext::intel::experimental::esimd;
using namespace cl::sycl;

void kernel0(uint32_t *ptr) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);

  atomic_update<atomic_op::inc, uint32_t, 32>(ptr, offsets, 1);
  // deprecated form:
  flat_atomic<EsimdAtomicOpType::ATOMIC_INC, uint32_t, 32>(ptr, offsets, 1);
}

void kernel1(uint32_t *ptr) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);
  simd<uint32_t, 32> v1(0, 1);

  atomic_update<atomic_op::add, uint32_t, 32>(ptr, offsets, v1, 1);
  // deprecated form:
  flat_atomic<EsimdAtomicOpType::ATOMIC_ADD, uint32_t, 32>(ptr, offsets, v1, 1);
}

void kernel2(uint32_t *ptr) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);
  simd<uint32_t, 32> v1(0, 1);

  atomic_update<atomic_op::cmpxchg, uint32_t, 32>(ptr, offsets, v1, v1, 1);
  // deprecated form:
  flat_atomic<EsimdAtomicOpType::ATOMIC_CMPXCHG, uint32_t, 32>(ptr, offsets, v1,
                                                               v1, 1);
}
