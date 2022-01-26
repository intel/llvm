// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// `Group algorithms are not supported on host device.` on Nvidia.
// XFAIL: hip_nvidia
//
// UNSUPPORTED: ze_debug-1,ze_debug4

// RUNx: %HOST_RUN_PLACEHOLDER %t.out
// TODO: Enable the test for HOST when it supports ext::oneapi::reduce() and
// barrier()

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// with reductions initialized with a placeholder accessor.

#include "reduction_utils.hpp"

using namespace cl::sycl;

template <typename T, int N> class KName;

template <typename Name, typename T, int Dim, class BinaryOperation,
          access::mode Mode>
int test(queue &Q, T Identity, T Init, size_t WGSize, size_t NWItems) {
  nd_range<1> NDRange(range<1>{NWItems}, range<1>{WGSize});
  printTestLabel<T, BinaryOperation>(false, NDRange);

  // Initialize.
  T CorrectOut;
  BinaryOperation BOp;

  buffer<T, 1> OutBuf(1);
  buffer<T, 1> InBuf(NWItems);
  initInputData(InBuf, CorrectOut, Identity, BOp, NWItems);
  if (Mode == access::mode::read_write)
    CorrectOut = BOp(CorrectOut, Init);

  (OutBuf.template get_access<access::mode::write>())[0] = Init;

  auto Out = accessor<T, Dim, Mode, access::target::device,
                      access::placeholder::true_t>(OutBuf);
  // Compute.
  Q.submit([&](handler &CGH) {
    auto In = InBuf.template get_access<access::mode::read>(CGH);
    CGH.require(Out);
    auto Redu = ext::oneapi::reduction(Out, Identity, BOp);
    CGH.parallel_for<Name>(NDRange, Redu, [=](nd_item<1> NDIt, auto &Sum) {
      Sum.combine(In[NDIt.get_global_linear_id()]);
    });
  });
  Q.wait();

  // Check correctness.
  T ReduVar = (OutBuf.template get_access<access::mode::read>())[0];
  return checkResults(Q, false /*SYCL2020*/, BOp, NDRange, ReduVar, CorrectOut);
}

int NumErrors = 0;

template <typename Name, typename T, class BinaryOperation>
void tests(queue &Q, T Identity, T Init, size_t WGSize, size_t NWItems) {
  constexpr access::mode DW = access::mode::discard_write;
  constexpr access::mode RW = access::mode::read_write;

  NumErrors += test<KName<Name, 0>, T, 0, BinaryOperation, DW>(
      Q, Identity, Init, WGSize, NWItems);
  NumErrors += test<KName<Name, 1>, T, 1, BinaryOperation, DW>(
      Q, Identity, Init, WGSize, NWItems);

  NumErrors += test<KName<Name, 2>, T, 0, BinaryOperation, RW>(
      Q, Identity, Init, WGSize, NWItems);
  NumErrors += test<KName<Name, 3>, T, 1, BinaryOperation, RW>(
      Q, Identity, Init, WGSize, NWItems);
}

int main() {
  queue Q;
  printDeviceInfo(Q);

  // fast atomics and fast reduce
  tests<class AtomicReduce, int, std::plus<>>(Q, 0, 77, 49, 49 * 5);

  // fast atomics
  tests<class Atomic, int, std::bit_or<>>(Q, 0, 233, 7, 7 * 3);

  // fast reduce
  tests<class Reduce, float, ext::oneapi::minimum<>>(
      Q, getMaximumFPValue<float>(), -5.0, 5, 5 * 7);

  // generic algorithm
  tests<class Generic, CustomVec<short>, CustomVecPlus<short>>(
      Q, CustomVec<short>(0), CustomVec<short>(4), 8, 8 * 3);

  printFinalStatus(NumErrors);
  return NumErrors;
}
