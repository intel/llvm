// RUN: %clangxx -fsycl -fsyntax-only -Wno-unused-command-line-argument %s

// This test checks compilation of ESIMD slm atomic APIs.

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>
#include <limits>
#include <utility>

using namespace sycl::ext::intel::experimental::esimd;
using namespace cl::sycl;

void kernel0() SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);

  slm_atomic_update<atomic_op::inc, uint32_t, 32>(offsets, 1);
}

void kernel1() SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);
  simd<uint32_t, 32> v1(0, 1);

  slm_atomic_update<atomic_op::add, uint32_t, 32>(offsets, v1, 1);
}

void kernel2() SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);
  simd<uint32_t, 32> v1(0, 1);

  slm_atomic_update<atomic_op::cmpxchg, uint32_t, 32>(offsets, v1, v1, 1);
}
