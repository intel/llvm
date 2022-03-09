// RUN: %clangxx -fsycl -fsyntax-only -Wno-unused-command-line-argument %s

// This test checks compilation of ESIMD atomic APIs.

#include <CL/sycl.hpp>
#include <limits>
#include <sycl/ext/intel/esimd.hpp>
#include <utility>

using namespace sycl::ext::intel::esimd;
using namespace cl::sycl;

void kernel0(uint32_t *ptr) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);

  atomic_update<atomic_op::inc, uint32_t, 32>(ptr, offsets, 1);
}

void kernel1(uint32_t *ptr) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);
  simd<uint32_t, 32> v1(0, 1);

  atomic_update<atomic_op::add, uint32_t, 32>(ptr, offsets, v1, 1);
}

template <class T> void kernel2(T *ptr) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);
  simd<T, 32> v1(0, 1);

  atomic_update<atomic_op::cmpxchg, T, 32>(ptr, offsets, v1, v1, 1);
}

template void kernel2<uint32_t>(uint32_t *) SYCL_ESIMD_FUNCTION;
