// REQUIRES: gpu
// RUN: %{build} -D__SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__ -o %t1.out

// RUN: env SYCL_PARALLEL_FOR_RANGE_ROUNDING_TRACE=1 %{run} %t1.out | FileCheck %s --check-prefix CHECK-DISABLED

// RUN: %{build} -sycl-std=2020 -o %t2.out
// RUN: env SYCL_PARALLEL_FOR_RANGE_ROUNDING_TRACE=1 %{run} %t2.out | FileCheck %s --check-prefix CHECK-ENABLED

#include <sycl/atomic_ref.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

#include "../helpers.hpp"

#include <iostream>
using namespace sycl;

void check(const char *msg, size_t v, size_t ref) {
  std::cout << msg << v << std::endl;
  assert(v == ref);
}

void try_rounding_off(size_t size, bool useShortcutFunction) {
  range<1> Range{size};
  queue Queue;
  range<1> *RangePtr = malloc_shared<range<1>>(1, Queue);
  int *CounterPtr = malloc_shared<int>(1, Queue);

  std::cout << "Run parallel_for" << std::endl;
  auto KernelFunc = [=](item<1> id) {
    auto atm = atomic_ref<int, sycl::memory_order::relaxed,
                          sycl::memory_scope::device>(*CounterPtr);
    atm.fetch_add(1);
    (*RangePtr) = id.get_range(0);
  };
  command_submit_wrappers::parallel_for_wrapper<class TestKernel>(
      useShortcutFunction, Queue, Range, KernelFunc);

  Queue.wait();

  auto Context = Queue.get_context();

  check("Size seen by user = ", RangePtr->get(0), size);
  check("Counter = ", *CounterPtr, size);

  free(RangePtr, Context);
  free(CounterPtr, Context);
}

int main() {
  int x;

  x = 1500;
  try_rounding_off(x, true);
  try_rounding_off(x, false);

  return 0;
}

// CHECK-DISABLED:  Run parallel_for
// CHECK-DISABLED-NOT: parallel_for range adjusted at dim 0 from 1500
// CHECK-DISABLED:  Size seen by user = 1500
// CHECK-DISABLED-NEXT:  Counter = 1500
// CHECK-DISABLED:  Run parallel_for
// CHECK-DISABLED-NOT: parallel_for range adjusted at dim 0 from 1500
// CHECK-DISABLED:  Size seen by user = 1500
// CHECK-DISABLED-NEXT:  Counter = 1500

// CHECK-ENABLED:  Run parallel_for
// CHECK-ENABLED-NEXT: parallel_for range adjusted at dim 0 from 1500
// CHECK-ENABLED-NEXT:  Size seen by user = 1500
// CHECK-ENABLED-NEXT:  Counter = 1500
// CHECK-ENABLED:  Run parallel_for
// CHECK-ENABLED-NEXT: parallel_for range adjusted at dim 0 from 1500
// CHECK-ENABLED-NEXT:  Size seen by user = 1500
// CHECK-ENABLED-NEXT:  Counter = 1500
