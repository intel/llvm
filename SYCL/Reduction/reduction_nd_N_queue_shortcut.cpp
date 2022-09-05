// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

// Group algorithms are not supported on NVidia.
// XFAIL: hip_nvidia

// This test only checks that the method queue::parallel_for() accepting
// reduction, can be properly translated into queue::submit + parallel_for().

#include "reduction_utils.hpp"

using namespace sycl;

enum TestCase { NoDependencies, Dependency, DependenciesVector };

template <typename T> T *allocUSM(queue &Q, size_t Size) {
  if (!Q.get_device().has(getUSMAspect(usm::alloc::shared)))
    return nullptr;

  return malloc_shared<T>(Size, Q);
}

template <typename RangeT>
void printNVarsTestLabel(const RangeT &Range, bool ToCERR = false) {
  std::ostream &OS = ToCERR ? std::cerr : std::cout;
  OS << (ToCERR ? "Error" : "Start") << ", Range=" << Range;
  if (!ToCERR)
    OS << std::endl;
}

template <typename T1, typename T2, TestCase TC, int Dims, typename BOpT1,
          typename BOpT2>
int test(queue &Q, BOpT1 BOp1, BOpT2 BOp2, const nd_range<Dims> &Range) {
  printNVarsTestLabel(Range);

  size_t NElems = Range.get_global_range().size();
  T1 *Sum1 = allocUSM<T1>(Q, 1);
  T2 *Sum2 = allocUSM<T2>(Q, 1);
  T1 *Arr1 = allocUSM<T1>(Q, NElems);
  T2 *Arr2 = allocUSM<T2>(Q, NElems);
  if (!Sum1 || !Sum2 || !Arr1 || !Arr2) {
    sycl::free(Sum1, Q);
    sycl::free(Sum2, Q);
    sycl::free(Arr1, Q);
    sycl::free(Arr2, Q);
    std::cout << " SKIPPED due to unrelated problems with USM" << std::endl;
    return 0;
  }

  *Sum2 = 0;
  auto R1 = sycl::reduction(
      Sum1, BOp1, property_list(property::reduction::initialize_to_identity{}));
  auto R2 = sycl::reduction(Sum2, static_cast<T2>(0), BOp2);

  if constexpr (TC == TestCase::NoDependencies) {
    std::fill(Arr1, Arr1 + NElems, 1);
    std::fill(Arr2, Arr2 + NElems, 2);
    Q.parallel_for(Range, R1, R2,
                   [=](nd_item<Dims> It, auto &Sum1, auto &Sum2) {
                     size_t LinId = It.get_global_linear_id();
                     Sum1.combine(static_cast<T1>(LinId) + Arr1[LinId]);
                     Sum2.combine(static_cast<T2>(LinId) + Arr2[LinId]);
                   })
        .wait();
  } else if constexpr (TC == TestCase::Dependency) {
    auto E = Q.single_task([=]() {
      std::fill(Arr1, Arr1 + NElems, 1);
      std::fill(Arr2, Arr2 + NElems, 2);
    });
    Q.parallel_for(Range, E, R1, R2,
                   [=](nd_item<Dims> It, auto &Sum1, auto &Sum2) {
                     size_t LinId = It.get_global_linear_id();
                     Sum1.combine(static_cast<T1>(LinId) + Arr1[LinId]);
                     Sum2.combine(static_cast<T2>(LinId) + Arr2[LinId]);
                   })
        .wait();
  } else {
    auto E1 = Q.single_task([=]() { std::fill(Arr1, Arr1 + NElems, 1); });
    auto E2 = Q.single_task([=]() { std::fill(Arr2, Arr2 + NElems, 2); });
    std::vector<event> EVec{E1, E2};
    Q.parallel_for(Range, EVec, R1, R2,
                   [=](nd_item<Dims> It, auto &Sum1, auto &Sum2) {
                     size_t LinId = It.get_global_linear_id();
                     Sum1.combine(static_cast<T1>(LinId) + Arr1[LinId]);
                     Sum2.combine(static_cast<T2>(LinId) + Arr2[LinId]);
                   })
        .wait();
  }

  T1 ExpectedSum1 = NElems + (NElems - 1) * NElems / 2;
  T2 ExpectedSum2 = 2 * NElems + (NElems - 1) * NElems / 2;
  std::string AddInfo = "TestCase=";
  int Error = checkResults(Q, BOp1, Range, *Sum1, ExpectedSum1,
                           AddInfo + std::to_string(1));
  Error += checkResults(Q, BOp2, Range, *Sum2, ExpectedSum2,
                        AddInfo + std::to_string(2));

  sycl::free(Sum1, Q);
  sycl::free(Sum2, Q);
  sycl::free(Arr1, Q);
  sycl::free(Arr2, Q);
  return Error;
}

template <typename T1, typename T2, int Dims, typename BinaryOperation1,
          typename BinaryOperation2>
int tests(queue &Q, BinaryOperation1 BOp1, BinaryOperation2 BOp2,
          const nd_range<Dims> &Range) {
  int NumErrors = 0;
  NumErrors += test<T1, T2, TestCase::NoDependencies>(Q, BOp1, BOp2, Range);
  NumErrors += test<T1, T2, TestCase::Dependency>(Q, BOp1, BOp2, Range);
  NumErrors += test<T1, T2, TestCase::DependenciesVector>(Q, BOp1, BOp2, Range);
  return NumErrors;
}

int main() {
  queue Q;
  printDeviceInfo(Q);

  int NumErrors = 0;
  auto LambdaSum = [](auto X, auto Y) { return (X + Y); };

  NumErrors +=
      tests<int, short>(Q, std::plus<>{}, LambdaSum, nd_range<1>{32, 16});
  NumErrors += tests<int, short>(Q, std::plus<>{}, LambdaSum,
                                 nd_range<2>{range<2>{4, 4}, range<2>{2, 2}});
  NumErrors +=
      tests<int, short>(Q, std::plus<>{}, LambdaSum,
                        nd_range<3>{range<3>{4, 4, 3}, range<3>{1, 2, 3}});

  printFinalStatus(NumErrors);
  return NumErrors;
}
