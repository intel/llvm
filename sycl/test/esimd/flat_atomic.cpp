// RUN: %clangxx -fsycl -fsycl-device-only -fsyntax-only -Xclang -verify %s
// expected-no-diagnostics

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/esimd.hpp>
#include <limits>
#include <utility>

using namespace sycl::ext::intel::experimental::esimd;
using namespace cl::sycl;

void kernel0(accessor<uint32_t, 1, access::mode::read_write, access::target::global_buffer> &buf) __attribute__((sycl_device)) {
  simd<uint32_t, 32> offsets(0, 1);

  flat_atomic<EsimdAtomicOpType::ATOMIC_INC, uint32_t, 32>(buf.get_pointer(), offsets, 1);
}

void kernel1(accessor<uint32_t, 1, access::mode::read_write, access::target::global_buffer> &buf) __attribute__((sycl_device)) {
  simd<uint32_t, 32> offsets(0, 1);
  simd<uint32_t, 32> v1(0, 1);

  flat_atomic<EsimdAtomicOpType::ATOMIC_ADD, uint32_t, 32>(buf.get_pointer(), offsets, v1, 1);
}

void kernel2(accessor<uint32_t, 1, access::mode::read_write, access::target::global_buffer> &buf) __attribute__((sycl_device)) {
  simd<uint32_t, 32> offsets(0, 1);
  simd<uint32_t, 32> v1(0, 1);

  flat_atomic<EsimdAtomicOpType::ATOMIC_CMPXCHG, uint32_t, 32>(buf.get_pointer(), offsets, v1, v1, 1);
}
