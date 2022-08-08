// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// `Group algorithms are not supported on host device` on Nvidia.
// XFAIL: hip_nvidia

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// with reductions initialized with USM var. It tests only SYCL-2020 reduction
// (sycl::reduction) assuming discard-write access, i.e. when reduction
// is created with property::reduction::initialize_to_identity.

#include "reduction_utils.hpp"

using namespace sycl;

template <typename T1, typename T2> class KName;

// Discard-write access to USM reductions is available only in SYCL2020.
constexpr bool IsSYCL2020 = true;

template <typename Name, typename T, class BinaryOperation>
int test(queue &Q, T Identity, T Init, size_t WGSize, size_t NWItems,
         usm::alloc AllocType) {
  nd_range<1> NDRange(range<1>{NWItems}, range<1>{WGSize});
  printTestLabel<T, BinaryOperation>(IsSYCL2020, NDRange);

  auto Dev = Q.get_device();
  if (!Dev.has(getUSMAspect(AllocType)))
    return 0;

  T *ReduVarPtr = (T *)malloc(sizeof(T), Dev, Q.get_context(), AllocType);
  if (ReduVarPtr == nullptr)
    return 0;
  if (AllocType == usm::alloc::device) {
    Q.submit([&](handler &CGH) {
       CGH.single_task<KName<Name, class InitKernel>>(
           [=]() { *ReduVarPtr = Init; });
     }).wait();
  } else {
    *ReduVarPtr = Init;
  }

  // Initialize.
  T CorrectOut;
  BinaryOperation BOp;

  buffer<T, 1> InBuf(NWItems);
  initInputData(InBuf, CorrectOut, Identity, BOp, NWItems);

  // Compute.
  Q.submit([&](handler &CGH) {
     auto In = InBuf.template get_access<access::mode::read>(CGH);
     auto Redu = createReduction<IsSYCL2020, access::mode::discard_write>(
         ReduVarPtr, Identity, BOp);
     CGH.parallel_for<KName<Name, class Test>>(
         NDRange, Redu, [=](nd_item<1> NDIt, auto &Sum) {
           Sum.combine(In[NDIt.get_global_linear_id()]);
         });
   }).wait();

  // Check correctness.
  T ComputedOut;
  if (AllocType == usm::alloc::device) {
    buffer<T, 1> Buf(&ComputedOut, range<1>(1));
    Q.submit([&](handler &CGH) {
       auto OutAcc = Buf.template get_access<access::mode::discard_write>(CGH);
       CGH.single_task<KName<Name, class Check>>(
           [=]() { OutAcc[0] = *ReduVarPtr; });
     }).wait();
    ComputedOut = (Buf.template get_access<access::mode::read>())[0];
  } else {
    ComputedOut = *ReduVarPtr;
  }
  std::string AllocStr =
      "AllocMode=" + std::to_string(static_cast<int>(AllocType));
  int Error = checkResults(Q, IsSYCL2020, BOp, NDRange, ComputedOut, CorrectOut,
                           AllocStr);

  free(ReduVarPtr, Q.get_context());
  return Error;
}

int NumErrors = 0;

template <typename Name, typename T, class BinaryOperation>
void testUSM(queue &Q, T Identity, T Init, size_t WGSize, size_t NWItems) {
  NumErrors += test<KName<Name, class Shared2020>, T, BinaryOperation>(
      Q, Identity, Init, WGSize, NWItems, usm::alloc::shared);
  NumErrors += test<KName<Name, class Host2020>, T, BinaryOperation>(
      Q, Identity, Init, WGSize, NWItems, usm::alloc::host);
  NumErrors += test<KName<Name, class Device2020>, T, BinaryOperation>(
      Q, Identity, Init, WGSize, NWItems, usm::alloc::device);
}

int main() {
  queue Q;
  printDeviceInfo(Q);

  // fast atomics and fast reduce
  testUSM<class AtomicReduce1, int, std::plus<int>>(Q, 0, 99, 49, 5 * 49);

  // fast atomics
  testUSM<class Atomic1, int, std::bit_or<>>(Q, 0, 0xff00ff00, 7, 7);
  testUSM<class Atomic2, int, std::bit_or<>>(Q, 0, 0x7f007f00, 4, 32);

  // fast reduce
  testUSM<class Reduce1, float, ext::oneapi::minimum<>>(
      Q, getMaximumFPValue<float>(), -100.0, 17, 17);
  testUSM<class Reduce2, float, ext::oneapi::maximum<>>(
      Q, getMinimumFPValue<float>(), 100.0, 4, 32);

  // generic algorithm
  testUSM<class Generic1, int, std::multiplies<>>(Q, 1, 5, 7, 7);
  testUSM<class Generic2, CustomVec<short>, CustomVecPlus<short>>(
      Q, CustomVec<short>(0), CustomVec<short>(77), 8, 8 * 3);

  printFinalStatus(NumErrors);
  return NumErrors;
}
