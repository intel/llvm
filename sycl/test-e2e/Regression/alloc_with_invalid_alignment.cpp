// REQUIRES: level_zero
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <algorithm>
#include <cassert>
#include <sycl/sycl.hpp>

// This test verifies that for L0 backend, aligned USM alloc functions return
// null_ptr when called with alignment values that are not a power-of-2.

using namespace sycl;

const size_t numBytes = 10;

void allocate_device(size_t alignment) {
  sycl::queue q;
  assert(!aligned_alloc_device(alignment, numBytes, q));
}

void allocate_shared(size_t alignment) {
  sycl::queue q;
  assert(!aligned_alloc_shared(alignment, numBytes, q));
}

void allocate_host(size_t alignment) {
  sycl::queue q;
  assert(!aligned_alloc_host(alignment, numBytes, q));
}

int main() {
  constexpr size_t alignmentCount = 20;
  size_t alignments[alignmentCount] = {3,  5,  6,   7,    9,    10,  12,
                                       15, 17, 18,  24,   30,   31,  33,
                                       63, 65, 100, 1023, 2049, 2050};
  std::for_each(alignments, alignments + alignmentCount - 1, allocate_device);
  std::for_each(alignments, alignments + alignmentCount - 1, allocate_shared);
  std::for_each(alignments, alignments + alignmentCount - 1, allocate_host);
  return 0;
}
