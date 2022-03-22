// REQUIRES: level_zero, level_zero_dev_kit
// TODO: ZE_DEBUG=4 produces no output on Windows. Enable when fixed.
// UNSUPPORTED: windows
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: env SYCL_DEVICE_FILTER=level_zero ZE_DEBUG=4 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER
//
// CHECK-NOT: LEAK

// Tests that additional resources required by USM reductions do not leak.

#include <CL/sycl.hpp>

using namespace cl::sycl;

int main() {
  queue Q;

  nd_range<1> NDRange(range<1>{49 * 5}, range<1>{49});
  std::plus<> BOp;

  int *Out = malloc_shared<int>(1, Q);
  int *In = malloc_shared<int>(49 * 5, Q);
  Q.submit([&](handler &CGH) {
     auto Redu = ext::oneapi::reduction(Out, 0, BOp);
     CGH.parallel_for<class USMSum>(
         NDRange, Redu, [=](nd_item<1> NDIt, auto &Sum) {
           Sum.combine(In[NDIt.get_global_linear_id()]);
         });
   }).wait();
  sycl::free(In, Q);
  sycl::free(Out, Q);
  return 0;
}
