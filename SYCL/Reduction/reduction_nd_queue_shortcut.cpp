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

template <typename T, TestCase TC, int Dims, typename BinaryOperation>
int test(queue &Q, BinaryOperation BOp, const nd_range<Dims> &Range) {
  printTestLabel<T, BinaryOperation>(Range);

  size_t NElems = Range.get_global_range().size();
  T *Sum = allocUSM<T>(Q, 1);
  T *Arr = allocUSM<T>(Q, NElems);
  if (!Sum || !Arr) {
    std::cout << " SKIPPED due to unrelated problems with USM" << std::endl;
    sycl::free(Sum, Q);
    sycl::free(Arr, Q);
    return 0;
  }

  auto Redu = sycl::reduction(
      Sum, BOp, property_list(property::reduction::initialize_to_identity{}));
  if constexpr (TC == TestCase::NoDependencies) {
    std::fill(Arr, Arr + NElems, 1);
    Q.parallel_for(Range, Redu, [=](nd_item<Dims> It, auto &Sum) {
       size_t LinId = It.get_global_linear_id();
       Sum.combine(static_cast<T>(LinId) + Arr[LinId]);
     }).wait();
  } else if constexpr (TC == TestCase::Dependency) {
    auto E = Q.single_task([=]() { std::fill(Arr, Arr + NElems, 1); });
    Q.parallel_for(Range, E, Redu, [=](nd_item<Dims> It, auto &Sum) {
       size_t LinId = It.get_global_linear_id();
       Sum.combine(static_cast<T>(LinId) + Arr[LinId]);
     }).wait();
  } else {
    auto E = Q.single_task([=]() { std::fill(Arr, Arr + NElems, 1); });
    std::vector<event> EVec{E};
    Q.parallel_for(Range, EVec, Redu, [=](nd_item<Dims> It, auto &Sum) {
       size_t LinId = It.get_global_linear_id();
       Sum.combine(static_cast<T>(LinId) + Arr[LinId]);
     }).wait();
  }

  T ExpectedSum = NElems + (NElems - 1) * NElems / 2;
  int Error = checkResults(Q, BOp, Range, *Sum, ExpectedSum);
  free(Sum, Q);
  free(Arr, Q);
  return Error;
}

template <typename T, int Dims, typename BinaryOperation>
int tests(queue &Q, BinaryOperation BOp, const nd_range<Dims> &Range) {
  int NumErrors = 0;
  NumErrors += test<T, TestCase::NoDependencies>(Q, BOp, Range);
  NumErrors += test<T, TestCase::Dependency>(Q, BOp, Range);
  NumErrors += test<T, TestCase::DependenciesVector>(Q, BOp, Range);
  return NumErrors;
}

int main() {
  queue Q;
  printDeviceInfo(Q);

  int NumErrors = 0;
  NumErrors += tests<int>(Q, std::plus<>{}, nd_range<1>{32, 16});
  NumErrors +=
      tests<int>(Q, std::plus<>{}, nd_range<2>{range<2>{4, 4}, range<2>{2, 2}});
  NumErrors += tests<int>(Q, std::plus<>{},
                          nd_range<3>{range<3>{4, 4, 3}, range<3>{1, 2, 3}});

  printFinalStatus(NumErrors);
  return NumErrors;
}
