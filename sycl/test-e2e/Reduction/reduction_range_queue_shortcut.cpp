// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Group algorithms are not supported on NVidia.
// XFAIL: hip_nvidia

// Windows doesn't yet have full shutdown().
// UNSUPPORTED: ze_debug && windows

// This test only checks that the shortcut method queue::parallel_for()
// can accept 2 or more reduction variables.

#include "reduction_utils.hpp"

using namespace sycl;

enum TestCase { NoDependencies, Dependency, DependenciesVector };

template <typename T> T *allocUSM(queue &Q, size_t Size) {
  if (!Q.get_device().has(getUSMAspect(usm::alloc::shared)))
    return nullptr;

  return malloc_shared<T>(Size, Q);
}

size_t linearizeId(id<1> Id, range<1>) { return Id; }
size_t linearizeId(id<2> Id, range<2> Range) {
  return Id[0] * Range[1] + Id[1];
}
size_t linearizeId(id<3> Id, range<3> Range) {
  return Id[0] * Range[1] * Range[2] + Id[1] * Range[2] + Id[2];
}

template <typename T, TestCase TC, int Dims, typename BinaryOperation>
int test(queue &Q, BinaryOperation BOp, const range<Dims> &Range) {
  printTestLabel<T, BinaryOperation>(Range);

  size_t NElems = Range.size();
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
    Q.parallel_for(Range, Redu, [=](id<Dims> Id, auto &Sum) {
       size_t LinId = linearizeId(Id, Range);
       Sum.combine(static_cast<T>(LinId) + Arr[LinId]);
     }).wait();
  } else if constexpr (TC == TestCase::Dependency) {
    auto E = Q.fill<T>(Arr, 1, NElems);
    Q.parallel_for(Range, E, Redu, [=](id<Dims> Id, auto &Sum) {
       size_t LinId = linearizeId(Id, Range);
       Sum.combine(static_cast<T>(LinId) + Arr[LinId]);
     }).wait();
  } else {
    auto E = Q.fill<T>(Arr, 1, NElems);
    std::vector<event> EVec{E};
    Q.parallel_for(Range, EVec, Redu, [=](id<Dims> Id, auto &Sum) {
       size_t LinId = linearizeId(Id, Range);
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
int tests(queue &Q, BinaryOperation BOp, const range<Dims> &Range) {
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
  NumErrors += tests<int>(Q, std::plus<>{}, range<1>{32});
  NumErrors += tests<int>(Q, std::plus<>{}, range<2>{4, 4});
  NumErrors += tests<int>(Q, std::plus<>{}, range<3>{4, 4, 3});

  printFinalStatus(NumErrors);
  return NumErrors;
}
