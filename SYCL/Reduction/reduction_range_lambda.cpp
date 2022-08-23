// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test performs basic checks of parallel_for(range, reduction, lambda)
// with reductions initialized with a one element buffer.

#include "reduction_utils.hpp"

using namespace sycl;

int main() {
  queue Q;
  printDeviceInfo(Q);
  size_t MaxWGSize =
      Q.get_device().get_info<info::device::max_work_group_size>();

  auto LambdaSum = [](auto x, auto y) { return (x + y); };

  int NumErrors = 0;

  NumErrors += test<class A1, int>(Q, 0, 99, LambdaSum, range<1>{7});
  NumErrors +=
      test<class A2, int>(Q, 0, 99, LambdaSum, range<1>{7}, init_to_identity());

  NumErrors +=
      test<class A3, int>(Q, 0, 99, LambdaSum, range<1>{MaxWGSize + 1});
  NumErrors += test<class A4, int>(Q, 0, 99, LambdaSum, range<1>{MaxWGSize + 1},
                                   init_to_identity());

  NumErrors += test<class B1, int>(Q, 0, 99, LambdaSum, range<2>{3, 4});
  NumErrors += test<class B2, int>(Q, 0, 99, LambdaSum, range<2>{3, 4},
                                   init_to_identity());

  NumErrors +=
      test<class B3, int>(Q, 0, 99, LambdaSum, range<2>{3, MaxWGSize + 1});
  NumErrors += test<class B4, int>(
      Q, 0, 99, LambdaSum, range<2>{3, MaxWGSize + 1}, init_to_identity());

  NumErrors += test<class C1, int>(Q, 0, 99, LambdaSum, range<3>{2, 3, 4});
  NumErrors += test<class C2, int>(Q, 0, 99, LambdaSum, range<3>{2, 3, 4},
                                   init_to_identity());

  NumErrors +=
      test<class C3, int>(Q, 0, 99, LambdaSum, range<3>{2, 3, MaxWGSize + 1});
  NumErrors += test<class C4, int>(
      Q, 0, 99, LambdaSum, range<3>{2, 3, MaxWGSize + 1}, init_to_identity());

  printFinalStatus(NumErrors);
  return NumErrors;
}
