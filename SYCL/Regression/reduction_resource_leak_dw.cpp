// REQUIRES: level_zero, level_zero_dev_kit
// TODO: ZE_DEBUG=4 produces no output on Windows. Enable when fixed.
// UNSUPPORTED: windows
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.out
// RUN: env SYCL_DEVICE_FILTER=level_zero ZE_DEBUG=4 %GPU_RUN_PLACEHOLDER %t.out 2>&1 %GPU_CHECK_PLACEHOLDER
//
// CHECK-NOT: LEAK

// Tests that additional resources required by discard_write reductions do not
// leak.

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue Q;

  nd_range<1> NDRange(range<1>{49 * 5}, range<1>{49});
  std::plus<> BOp;

  buffer<int, 1> OutBuf(1);
  buffer<int, 1> InBuf(49 * 5);
  Q.submit([&](handler &CGH) {
    auto In = InBuf.get_access<access::mode::read>(CGH);
    auto Out = OutBuf.get_access<access::mode::discard_write>(CGH);
    auto Redu = ext::oneapi::reduction(Out, 0, BOp);
    CGH.parallel_for<class DiscardSum>(
        NDRange, Redu, [=](nd_item<1> NDIt, auto &Sum) {
          Sum.combine(In[NDIt.get_global_linear_id()]);
        });
  });
  return 0;
}
