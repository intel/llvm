// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

// RUNx: %HOST_RUN_PLACEHOLDER %t.out
// TODO: Enable the test for HOST when it supports ONEAPI::reduce() and
// barrier()

// This test only checks that the method queue::parallel_for() accepting
// reduction, can be properly translated into queue::submit + parallel_for().

#include <CL/sycl.hpp>
using namespace sycl;

template <typename T, bool B> class KName;

template <typename Name, bool IsSYCL2020Mode> int test(queue &Q) {
  const size_t NElems = 1024;
  const size_t WGSize = 256;

  int *Data = malloc_shared<int>(NElems, Q);
  for (int I = 0; I < NElems; I++)
    Data[I] = I;

  int *Sum = malloc_shared<int>(1, Q);
  *Sum = 0;

  if constexpr (IsSYCL2020Mode) {
    Q.parallel_for<Name>(
         nd_range<1>{NElems, WGSize}, sycl::reduction(Sum, std::plus<>()),
         [=](nd_item<1> It, auto &Sum) { Sum += Data[It.get_global_id(0)]; })
        .wait();
  } else {
    Q.parallel_for<Name>(
         nd_range<1>{NElems, WGSize}, ONEAPI::reduction(Sum, ONEAPI::plus<>()),
         [=](nd_item<1> It, auto &Sum) { Sum += Data[It.get_global_id(0)]; })
        .wait();
  }

  int ExpectedSum = (NElems - 1) * NElems / 2;
  int Error = 0;
  if (*Sum != ExpectedSum) {
    std::cerr << "Error: Expected = " << ExpectedSum << ", Computed = " << *Sum
              << std::endl;
    Error = 1;
  }

  free(Data, Q);
  free(Sum, Q);
  return Error;
}

int main() {
  queue Q;
  int Error = test<KName<class A, true>, true>(Q);
  Error += test<KName<class A, false>, false>(Q);
  return Error;
}
