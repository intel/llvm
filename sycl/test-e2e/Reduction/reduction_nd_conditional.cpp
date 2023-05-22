// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
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
  host_accessor In(InBuf, write_only);
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
  printTestLabel<T, BinaryOperation>(NDRange);

  // Initialize.
  BinaryOperation BOp;
  T CorrectOut;
  initInputData(InBuf, CorrectOut, Identity, BOp, NWItems);
  host_accessor(OutBuf, write_only)[0] = Identity;

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
  host_accessor Out(OutBuf, read_only);
  T ComputedOut = *(Out.get_pointer());
  return checkResults(Q, BOp, NDRange, ComputedOut, CorrectOut);
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
