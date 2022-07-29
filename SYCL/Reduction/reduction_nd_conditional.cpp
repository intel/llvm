// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUNx: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// Error message `The implementation handling
// parallel_for with reduction requires work group size not bigger than 1` on
// Nvidia.
// XFAIL: hip_nvidia

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// with reduction and conditional increment of the reduction variable.

#include "reduction_utils.hpp"

using namespace sycl;

template <typename T, class BinaryOperation>
void initInputData(buffer<T, 1> &InBuf, T &ExpectedOut, T Identity,
                   BinaryOperation BOp, size_t N) {
  ExpectedOut = Identity;
  auto In = InBuf.template get_access<access::mode::write>();
  for (int I = 0; I < N; ++I) {
    In[I] = I + 1;

    if (I < 2)
      ExpectedOut = BOp(ExpectedOut, 99);
    else if (I % 3)
      ExpectedOut = BOp(ExpectedOut, In[I]);
    else
      ; // do nothing.
  }
};

template <typename Name, typename T, class BinaryOperation>
int test(queue &Q, T Identity, size_t WGSize, size_t NWItems) {
  buffer<T, 1> InBuf(NWItems);
  buffer<T, 1> OutBuf(1);

  nd_range<1> NDRange(range<1>{NWItems}, range<1>{WGSize});
  printTestLabel<T, BinaryOperation>(true /*SYCL2020*/, NDRange);

  // Initialize.
  BinaryOperation BOp;
  T CorrectOut;
  initInputData(InBuf, CorrectOut, Identity, BOp, NWItems);
  (OutBuf.template get_access<access::mode::write>())[0] = Identity;

  // Compute.
  Q.submit([&](handler &CGH) {
    auto In = InBuf.template get_access<access::mode::read>(CGH);
    auto Redu = sycl::reduction(OutBuf, CGH, Identity, BOp);
    CGH.parallel_for<Name>(NDRange, Redu, [=](nd_item<1> NDIt, auto &Sum) {
      size_t I = NDIt.get_global_linear_id();
      if (I < 2)
        Sum.combine(T(99));
      else if (I % 3)
        Sum.combine(In[I]);
      else
        ; // do nothing.
    });
  });

  // Check correctness.
  auto Out = OutBuf.template get_access<access::mode::read>();
  T ComputedOut = *(Out.get_pointer());
  return checkResults(Q, true /*SYCL2020*/, BOp, NDRange, ComputedOut,
                      CorrectOut);
}

int main() {
  queue Q;
  printDeviceInfo(Q);

  int NumErrors = 0;
  NumErrors += test<class A, unsigned char, std::plus<>>(Q, 0, 2, 2);
  NumErrors += test<class B, short, std::plus<>>(Q, 0, 7, 7);
  NumErrors += test<class C, int, std::plus<>>(Q, 0, 1, 1025);

  printFinalStatus(NumErrors);
  return NumErrors;
}
