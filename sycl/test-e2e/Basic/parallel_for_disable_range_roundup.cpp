// REQUIRES: gpu
// RUN: %{build} -D__SYCL_DISABLE_PARALLEL_FOR_RANGE_ROUNDING__ -o %t.out

// RUN: env SYCL_PARALLEL_FOR_RANGE_ROUNDING_TRACE=1 %{run} %t.out | FileCheck %s --check-prefix CHECK-DISABLED

// RUN: %{build} -sycl-std=2017 -o %t.out
// RUN: env SYCL_PARALLEL_FOR_RANGE_ROUNDING_TRACE=1 %{run} %t.out | FileCheck %s --check-prefix CHECK-DISABLED

// RUN: %{build} -sycl-std=2020 -o %t.out
// RUN: env SYCL_PARALLEL_FOR_RANGE_ROUNDING_TRACE=1 %{run} %t.out | FileCheck %s --check-prefix CHECK-ENABLED

#include <iostream>
#include <sycl/detail/core.hpp>
using namespace sycl;

range<1> Range1 = {0};

void check(const char *msg, size_t v, size_t ref) {
  std::cout << msg << v << std::endl;
  assert(v == ref);
}

int try_rounding_off(size_t size) {
  range<1> Size{size};
  int Counter = 0;
  {
    buffer<range<1>, 1> BufRange(&Range1, 1);
    buffer<int, 1> BufCounter(&Counter, 1);
    queue myQueue;

    std::cout << "Run parallel_for" << std::endl;
    myQueue.submit([&](handler &cgh) {
      auto AccRange = BufRange.get_access<access::mode::read_write>(cgh);
      auto AccCounter = BufCounter.get_access<access::mode::atomic>(cgh);
      cgh.parallel_for<class PF_init_item1>(Size, [=](item<1> ITEM) {
        AccCounter[0].fetch_add(1);
        AccRange[0] = ITEM.get_range(0);
      });
    });
    myQueue.wait();
  }
  check("Size seen by user = ", Range1.get(0), size);
  check("Counter = ", Counter, size);
  return 0;
}

int main() {
  int x;

  x = 1500;
  try_rounding_off(x);

  return 0;
}

// CHECK-DISABLED:  Run parallel_for
// CHECK-DISABLED-NOT: parallel_for range adjusted at dim 0 from 1500
// CHECK-DISABLED:  Size seen by user = 1500
// CHECK-DISABLED-NEXT:  Counter = 1500

// CHECK-ENABLED:  Run parallel_for
// CHECK-ENABLED-NEXT: parallel_for range adjusted at dim 0 from 1500
// CHECK-ENABLED-NEXT:  Size seen by user = 1500
// CHECK-ENABLED-NEXT:  Counter = 1500
