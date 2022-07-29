// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test performs basic checks of parallel_for(range, reduction, lambda)
// with reductions initialized with 1-dimensional accessor accessing
// 1 element buffer.

#include "reduction_range_scalar.hpp"

using namespace sycl;

constexpr access::mode RW = access::mode::read_write;
constexpr access::mode DW = access::mode::discard_write;

int main() {
  queue Q;
  printDeviceInfo(Q);
  size_t MaxWGSize =
      Q.get_device().get_info<info::device::max_work_group_size>();

  auto LambdaSum = [](auto x, auto y) { return (x + y); };

  int NumErrors = 0;

  NumErrors += testBoth<class A1, RW, int>(Q, 0, 99, LambdaSum, range<1>{7});
  NumErrors += testBoth<class A2, DW, int>(Q, 0, 99, LambdaSum, range<1>{7});

  NumErrors +=
      testBoth<class A3, RW, int>(Q, 0, 99, LambdaSum, range<1>{MaxWGSize + 1});
  NumErrors +=
      testBoth<class A4, DW, int>(Q, 0, 99, LambdaSum, range<1>{MaxWGSize + 1});

  NumErrors += testBoth<class B1, RW, int>(Q, 0, 99, LambdaSum, range<2>{3, 4});
  NumErrors += testBoth<class B2, DW, int>(Q, 0, 99, LambdaSum, range<2>{3, 4});

  NumErrors += testBoth<class B3, RW, int>(Q, 0, 99, LambdaSum,
                                           range<2>{3, MaxWGSize + 1});
  NumErrors += testBoth<class B4, DW, int>(Q, 0, 99, LambdaSum,
                                           range<2>{3, MaxWGSize + 1});

  NumErrors +=
      testBoth<class C1, RW, int>(Q, 0, 99, LambdaSum, range<3>{2, 3, 4});
  NumErrors +=
      testBoth<class C2, RW, int>(Q, 0, 99, LambdaSum, range<3>{2, 3, 4});

  NumErrors += testBoth<class C3, RW, int>(Q, 0, 99, LambdaSum,
                                           range<3>{2, 3, MaxWGSize + 1});
  NumErrors += testBoth<class C4, DW, int>(Q, 0, 99, LambdaSum,
                                           range<3>{2, 3, MaxWGSize + 1});

  printFinalStatus(NumErrors);
  return NumErrors;
}
