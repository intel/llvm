// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test performs basic checks of parallel_for(range, reduction, lambda)
// with reductions initialized with a one element buffer.

#include "reduction_utils.hpp"

using namespace sycl;

int NumErrors = 0;

template <typename Name, typename T, typename... ArgTys>
void tests(ArgTys &&...Args) {
  NumErrors += test<Name, T>(std::forward<ArgTys>(Args)...);
}

int main() {
  queue Q;
  printDeviceInfo(Q);
  size_t MaxWGSize =
      Q.get_device().get_info<info::device::max_work_group_size>();

  auto LambdaSum = [](auto x, auto y) { return (x + y); };

  tests<class A1, int>(Q, 0, 99, LambdaSum, range<1>{7});
  tests<class A2, int>(Q, 0, 99, LambdaSum, range<1>{7}, init_to_identity());
  tests<class A3, int>(Q, 99, LambdaSum, range<1>{7});

  tests<class A4, int>(Q, 0, 99, LambdaSum, range<1>{MaxWGSize + 1});
  tests<class A5, int>(Q, 0, 99, LambdaSum, range<1>{MaxWGSize + 1},
                       init_to_identity());
  tests<class A6, int>(Q, 99, LambdaSum, range<1>{MaxWGSize + 1});

  tests<class B1, int>(Q, 0, 99, LambdaSum, range<2>{3, 4});
  tests<class B2, int>(Q, 0, 99, LambdaSum, range<2>{3, 4}, init_to_identity());
  tests<class B3, int>(Q, 99, LambdaSum, range<2>{3, 4});

  tests<class B4, int>(Q, 0, 99, LambdaSum, range<2>{3, MaxWGSize + 1});
  tests<class B5, int>(Q, 0, 99, LambdaSum, range<2>{3, MaxWGSize + 1},
                       init_to_identity());
  tests<class B6, int>(Q, 99, LambdaSum, range<2>{3, MaxWGSize + 1});

  tests<class C1, int>(Q, 0, 99, LambdaSum, range<3>{2, 3, 4});
  tests<class C2, int>(Q, 0, 99, LambdaSum, range<3>{2, 3, 4},
                       init_to_identity());
  tests<class C3, int>(Q, 99, LambdaSum, range<3>{2, 3, 4});

  tests<class C4, int>(Q, 0, 99, LambdaSum, range<3>{2, 3, MaxWGSize + 1});
  tests<class C5, int>(Q, 0, 99, LambdaSum, range<3>{2, 3, MaxWGSize + 1},
                       init_to_identity());
  tests<class C6, int>(Q, 99, LambdaSum, range<3>{2, 3, MaxWGSize + 1});

  printFinalStatus(NumErrors);
  return NumErrors;
}
