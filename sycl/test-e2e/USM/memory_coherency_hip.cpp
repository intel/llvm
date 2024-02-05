// RUN: %{build} -o %t1.out
// REQUIRES: hip_amd
// RUN: %{run} %t1.out

//==---- memory_coherency_hip.cpp  -----------------------------------------==//
// USM coarse/fine grain memory coherency test for the HIP-AMD backend.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <bits/chrono.h>
#include <iostream>
#include <thread>

namespace kernels {
class SquareKrnl final {
  int *mPtr;

public:
  SquareKrnl(int *ptr) : mPtr{ptr} {}

  void operator()(sycl::id<1>) const { *mPtr = (*mPtr) * (*mPtr); }
};

class CoherencyTestKrnl final {
  int *mPtr;

public:
  CoherencyTestKrnl(int *ptr) : mPtr{ptr} {}

  void operator()(sycl::id<1>) const {
    auto atm = sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                sycl::memory_scope::device>(mPtr[0]);

    // mPtr was initialized to 1 by the host, now set it to 2.
    atm.fetch_add(1);

    // spin until mPtr is 3, then change it to 4.
    int expected{3};
    int old = atm.load();
    while (true) {
      old = atm.load();
      if (old == expected) {
        if (atm.compare_exchange_strong(old, 4)) {
          break;
        }
      }
    }
  }
};
} // namespace kernels

int main() {
  sycl::queue q{};
  sycl::device dev = q.get_device();
  sycl::context ctx = q.get_context();
  if (!dev.get_info<sycl::info::device::usm_shared_allocations>()) {
    std::cout << "Shared USM is not supported. Skipping test.\n";
    return 0;
  }

  bool coherent{false};

  int *ptr = sycl::malloc_shared<int>(1, q);

  // Coherency test 1
  //
  // The following test validates if memory access is fine with memory allocated
  // using malloc_managed() and COARSE_GRAINED advice set via mem_advise().
  //
  // Coarse grained memory is only guaranteed to be coherent outside of GPU
  // kernels that modify it. Changes applied to coarse-grained memory by a GPU
  // kernel are only visible to the rest of the system (CPU or other GPUs) when
  // the kernel has completed. A GPU kernel is only guaranteed to see changes
  // applied to coarse grained memory by the rest of the system (CPU or other
  // GPUs) if those changes were made before the kernel launched.

  // Hint to use coarse-grain memory.
  q.mem_advise(ptr, sizeof(int), int{PI_MEM_ADVICE_HIP_SET_COARSE_GRAINED});

  int init_val{9};
  int expected{init_val * init_val};

  *ptr = init_val;
  q.parallel_for(sycl::range{1}, kernels::SquareKrnl{ptr});
  // Synchronise the underlying stream.
  q.wait();

  // Check if caches are flushed correctly and same memory is between devices.
  if (*ptr == expected) {
    coherent = true;
  } else {
    std::cerr << "Coherency test failed. Value: " << *ptr
              << " (expected: " << expected << ")\n";
    coherent = false;
  }

  // Coherency test 2
  //
  // The following test validates if fine-grain behavior is observed or not with
  // memory allocated using malloc_managed().
  //
  // Fine grained memory allows CPUs and GPUs to synchronize (via atomics) and
  // coherently communicate with each other while the GPU kernel is running.

  // Hint to use fine-grain memory.
  q.mem_advise(ptr, sizeof(int), int{PI_MEM_ADVICE_HIP_UNSET_COARSE_GRAINED});

  init_val = 1;
  expected = 4;

  *ptr = init_val;
  q.parallel_for(sycl::range{1}, kernels::CoherencyTestKrnl{ptr});

  // wait until ptr is 2 from the kernel (or 3 seconds), then increment to 3.
  while (*ptr == 2) {
    using std::chrono_literals::operator""s;
    std::this_thread::sleep_for(3s);
    break;
  }
  *ptr += 1;

  // Synchronise the underlying stream.
  q.wait();

  // Check if caches are flushed correctly and same memory is between devices.
  if (*ptr == expected) {
    coherent &= true;
  } else {
    std::cerr << "Coherency test failed. Value: " << *ptr
              << " (expected: " << expected << ")\n";
    coherent = false;
  }

  // Cleanup
  sycl::free(ptr, q);

  // Check if all coherency tests passed.
  assert(coherent);
  // The above assert won't trigger with NDEBUG, so ensure the right exit code.
  return coherent ? 0 : 1;
}
