// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// SYCL 2020 (Work-group data parallel kernels): "When the global size is
// zero, the kernel function is not executed, the local size is ignored, and
// any dependencies are satisfied."
//
// See intel/llvm#22893 
//
// A USM-shared sentinel byte is set to 0 before each submit. If a kernel
// body actually ran, it would flip the byte to 0xFF. After Q.wait(), we
// assert the byte is still 0.

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include <cassert>

using namespace sycl;

int main() {
  queue Q;

  unsigned char *Sentinel = malloc_shared<unsigned char>(1, Q);
  assert(Sentinel && "USM shared alloc failed");

  // Case 1: parallel_for(range<1>{0}) -- plain empty range.
  *Sentinel = 0x00;
  Q.submit([&](handler &cgh) {
     cgh.parallel_for<class zero_range_pf>(
         range<1>{0}, [=](id<1>) { *Sentinel = 0xFF; });
   }).wait();
  assert(*Sentinel == 0x00 && "parallel_for(range{0}) unexpectedly launched");

  // Case 2: parallel_for(nd_range<1>{{0}, {32}}) -- the PyTorch shape:
  // zero global size, non-zero local size. Pre-fix, this tripped the
  // assertion in adjustNDRangePerKernel.
  *Sentinel = 0x00;
  Q.submit([&](handler &cgh) {
     cgh.parallel_for<class zero_range_ndr>(
         nd_range<1>{range<1>{0}, range<1>{32}},
         [=](nd_item<1>) { *Sentinel = 0xFF; });
   }).wait();
  assert(*Sentinel == 0x00 &&
         "parallel_for(nd_range{0, 32}) unexpectedly launched");

  free(Sentinel, Q);
  return 0;
}
