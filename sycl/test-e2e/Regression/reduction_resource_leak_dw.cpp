// REQUIRES: level_zero, level_zero_dev_kit
// XFAIL: windows
//
// RUN: %{build} %level_zero_options -o %t.out
// RUN: %{l0_leak_check} %{run} %t.out 2>&1 | FileCheck %s
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
    auto Redu = reduction(OutBuf, CGH, 0, BOp,
                          {property::reduction::initialize_to_identity{}});
    CGH.parallel_for<class DiscardSum>(
        NDRange, Redu, [=](nd_item<1> NDIt, auto &Sum) {
          Sum.combine(In[NDIt.get_global_linear_id()]);
        });
  });
  return 0;
}
