// REQUIRES: level_zero, level_zero_dev_kit
// TODO: UR_L0_LEAKS_DEBUG=1 produces no output on Windows. Enable when fixed.
// UNSUPPORTED: windows
//
// RUN: %{build} %level_zero_options -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR="level_zero:*" %{l0_leak_check}  %{run} %t.out 2>&1 | FileCheck %s
//
// CHECK-NOT: LEAK

// Tests that additional resources required by USM reductions do not leak.

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue Q;

  nd_range<1> NDRange(range<1>{49 * 5}, range<1>{49});
  std::plus<> BOp;

  int *Out = malloc_shared<int>(1, Q);
  int *In = malloc_shared<int>(49 * 5, Q);
  Q.submit([&](handler &CGH) {
     auto Redu = reduction(Out, 0, BOp);
     CGH.parallel_for<class USMSum>(
         NDRange, Redu, [=](nd_item<1> NDIt, auto &Sum) {
           Sum.combine(In[NDIt.get_global_linear_id()]);
         });
   }).wait();
  sycl::free(In, Q);
  sycl::free(Out, Q);
  return 0;
}
