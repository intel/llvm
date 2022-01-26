// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
//
// `Group algorithms are not supported on host device.` on Nvidia.
// XFAIL: hip_nvidia

// TODO: test disabled due to sporadic fails in level_zero:gpu RT.
// UNSUPPORTED: linux && level_zero

// RUNx: %HOST_RUN_PLACEHOLDER %t.out
// TODO: Enable the test for HOST when it supports ext::oneapi::reduce() and
// barrier()

// This test only checks that the method queue::parallel_for() accepting
// reduction, can be properly translated into queue::submit + parallel_for().

#include "reduction_utils.hpp"

using namespace sycl;

template <typename T, bool B> class KName;

template <typename Name, bool IsSYCL2020, typename T, typename BinaryOperation>
int test(queue &Q, T Identity, size_t WGSize, size_t NElems) {
  nd_range<1> NDRange(range<1>{NElems}, range<1>{WGSize});
  printTestLabel<T, BinaryOperation>(IsSYCL2020, NDRange);

  T *Data = malloc_shared<T>(NElems, Q);
  for (int I = 0; I < NElems; I++)
    Data[I] = I;

  T *Sum = malloc_shared<T>(1, Q);
  *Sum = Identity;

  BinaryOperation BOp;
  auto Redu = createReduction<IsSYCL2020, access::mode::read_write>(Sum, BOp);
  Q.parallel_for<Name>(NDRange, Redu, [=](nd_item<1> It, auto &Sum) {
     Sum += Data[It.get_global_id(0)];
   }).wait();

  T ExpectedSum = (NElems - 1) * NElems / 2;
  int Error = checkResults(Q, IsSYCL2020, BOp, NDRange, *Sum, ExpectedSum);

  free(Data, Q);
  free(Sum, Q);
  return Error;
}

int main() {
  queue Q;
  printDeviceInfo(Q);

  int NumErrors = test<class A1, true, int, std::plus<>>(Q, 0, 16, 32);
  NumErrors += test<class A2, false, int, std::plus<>>(Q, 0, 7, 14);

  printFinalStatus(NumErrors);
  return NumErrors;
}
