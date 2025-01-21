// RUN: not %clangxx -fsycl -fsycl-device-only -fsyntax-only -Wno-unused-command-line-argument %s 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"
// RUN: not %clangxx %fsycl-host-only -fsyntax-only -Wno-unused-command-line-argument %s 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

// This test checks compilation of ESIMD atomic APIs.

#include <limits>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>
#include <utility>

using namespace sycl::ext::intel::esimd;
using namespace sycl;

// --- Positive tests.

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

void kernel3(accessor<uint32_t, 1, access::mode::read_write,
                      access::target::device> &buf) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);

  atomic_update<atomic_op::inc, uint32_t, 32>(buf, offsets, 1);
}

void kernel4(accessor<uint32_t, 1, access::mode::read_write,
                      access::target::device> &buf) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);
  simd<uint32_t, 32> v1(0, 1);

  atomic_update<atomic_op::add, uint32_t, 32>(buf, offsets, v1, 1);
}

void kernel5(accessor<uint32_t, 1, access::mode::read_write,
                      access::target::device> &buf) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);
  simd<uint32_t, 32> v1(0, 1);

  atomic_update<atomic_op::cmpxchg, uint32_t, 32>(buf, offsets, v1, v1, 1);
}

void kernel6(local_accessor<uint32_t, 1> &buf) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);

  atomic_update<atomic_op::inc, uint32_t, 32>(buf, offsets, 1);
}

void kernel7(local_accessor<uint32_t, 1> &buf) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);
  simd<uint32_t, 32> v1(0, 1);

  atomic_update<atomic_op::add, uint32_t, 32>(buf, offsets, v1, 1);
}

void kernel8(local_accessor<uint32_t, 1> &buf) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);
  simd<uint32_t, 32> v1(0, 1);

  atomic_update<atomic_op::cmpxchg, uint32_t, 32>(buf, offsets, v1, v1, 1);
}

// --- Negative tests.

// Incompatible mode (read).
void kernel9(accessor<uint32_t, 1, access::mode::read, access::target::device>
                 &buf) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);

  // CHECK: flat_atomic.cpp:89{{.*}}error: no matching function for call to 'atomic_update'
  atomic_update<atomic_op::inc, uint32_t, 32>(buf, offsets, 1);
}

// Incompatible mode (read).
void kernel10(accessor<uint32_t, 1, access::mode::read, access::target::device>
                  &buf) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);
  simd<uint32_t, 32> v1(0, 1);

  // CHECK: flat_atomic.cpp:99{{.*}}error: no matching function for call to 'atomic_update'
  atomic_update<atomic_op::add, uint32_t, 32>(buf, offsets, v1, 1);
}

// Incompatible mode (read).
void kernel11(accessor<uint32_t, 1, access::mode::read, access::target::device>
                  &buf) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);
  simd<uint32_t, 32> v1(0, 1);

  // CHECK: flat_atomic.cpp:109{{.*}}error: no matching function for call to 'atomic_update'
  atomic_update<atomic_op::cmpxchg, uint32_t, 32>(buf, offsets, v1, v1, 1);
}

// Incompatible mode (read).
void kernel12(local_accessor<const uint32_t, 1> &buf) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);

  // CHECK: flat_atomic.cpp:117{{.*}}error: no matching function for call to 'atomic_update'
  atomic_update<atomic_op::inc, uint32_t, 32>(buf, offsets, 1);
}

// Incompatible mode (read).
void kernel13(local_accessor<const uint32_t, 1> &buf) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);
  simd<uint32_t, 32> v1(0, 1);

  // CHECK: flat_atomic.cpp:126{{.*}}error: no matching function for call to 'atomic_update'
  atomic_update<atomic_op::add, uint32_t, 32>(buf, offsets, v1, 1);
}

// Incompatible mode (read).
void kernel8(const local_accessor<const uint32_t, 1> &buf) SYCL_ESIMD_FUNCTION {
  simd<uint32_t, 32> offsets(0, 1);
  simd<uint32_t, 32> v1(0, 1);

  // CHECK: flat_atomic.cpp:135{{.*}}error: no matching function for call to 'atomic_update'
  atomic_update<atomic_op::cmpxchg, uint32_t, 32>(buf, offsets, v1, v1, 1);
}
