// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// TODO: this is a temporary solution until the odd performance effect
// on opencl:cpu is analyzed/fixed. Running 2x more test cases with USM
// reductions may cause 10x longer execution time right now.
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -DTEST_SYCL2020_REDUCTIONS %s -o %t2020.out
// RUN: %CPU_RUN_PLACEHOLDER %t2020.out
// RUN: %GPU_RUN_PLACEHOLDER %t2020.out
// RUN: %ACC_RUN_PLACEHOLDER %t2020.out

// RUNx: %HOST_RUN_PLACEHOLDER %t.out
// TODO: Enable the test for HOST when it supports ONEAPI::reduce() and
// barrier()

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// with reductions initialized with USM var.

// This test fails with exceeded time out on Windows with OpenCL, temporarily
// disabling
// UNSUPPORTED: windows && opencl

#include "reduction_utils.hpp"
#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

template <typename T1, typename T2> class KernelNameGroup;

template <bool IsSYCL2020Mode, typename T, typename BinaryOperation>
auto createReduction(T *USMPtr, T Identity, BinaryOperation BOp) {
  if constexpr (IsSYCL2020Mode)
    return sycl::reduction(USMPtr, Identity, BOp);
  else
    return ONEAPI::reduction(USMPtr, Identity, BOp);
}

template <typename Name, bool IsSYCL2020Mode, typename T, class BinaryOperation>
void test(T Identity, size_t WGSize, size_t NWItems, usm::alloc AllocType) {
  queue Q;
  auto Dev = Q.get_device();

  if (AllocType == usm::alloc::shared &&
      !Dev.get_info<info::device::usm_shared_allocations>())
    return;
  if (AllocType == usm::alloc::host &&
      !Dev.get_info<info::device::usm_host_allocations>())
    return;
  if (AllocType == usm::alloc::device &&
      !Dev.get_info<info::device::usm_device_allocations>())
    return;

  T *ReduVarPtr = (T *)malloc(sizeof(T), Dev, Q.get_context(), AllocType);
  if (ReduVarPtr == nullptr)
    return;
  if (AllocType == usm::alloc::device) {
    Q.submit([&](handler &CGH) {
       CGH.single_task<KernelNameGroup<Name, class Init>>(
           [=]() { *ReduVarPtr = Identity; });
     }).wait();
  } else {
    *ReduVarPtr = Identity;
  }

  // Initialize.
  T CorrectOut;
  BinaryOperation BOp;

  buffer<T, 1> InBuf(NWItems);
  initInputData(InBuf, CorrectOut, Identity, BOp, NWItems);

  // Compute.
  Q.submit([&](handler &CGH) {
     auto In = InBuf.template get_access<access::mode::read>(CGH);
     auto Redu = createReduction<IsSYCL2020Mode>(ReduVarPtr, Identity, BOp);
     nd_range<1> NDRange(range<1>{NWItems}, range<1>{WGSize});
     CGH.parallel_for<KernelNameGroup<Name, class Test>>(
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
       CGH.single_task<KernelNameGroup<Name, class Check>>(
           [=]() { OutAcc[0] = *ReduVarPtr; });
     }).wait();
    ComputedOut = (Buf.template get_access<access::mode::read>())[0];
  } else {
    ComputedOut = *ReduVarPtr;
  }
  if (ComputedOut != CorrectOut) {
    std::cout << "NWItems = " << NWItems << ", WGSize = " << WGSize << "\n";
    std::cout << "Computed value: " << ComputedOut
              << ", Expected value: " << CorrectOut
              << ", AllocMode: " << static_cast<int>(AllocType) << "\n";
    assert(0 && "Wrong value.");
  }

  free(ReduVarPtr, Q.get_context());
}

template <typename Name, typename T, class BinaryOperation>
void testUSM(T Identity, size_t WGSize, size_t NWItems) {
#ifdef TEST_SYCL2020_REDUCTIONS
  test<KernelNameGroup<Name, class SharedCase2020>, true, T, BinaryOperation>(
      Identity, WGSize, NWItems, usm::alloc::shared);
  test<KernelNameGroup<Name, class HostCase2020>, true, T, BinaryOperation>(
      Identity, WGSize, NWItems, usm::alloc::host);
  test<KernelNameGroup<Name, class DeviceCase2020>, true, T, BinaryOperation>(
      Identity, WGSize, NWItems, usm::alloc::device);
#else
  test<KernelNameGroup<Name, class SharedCase>, false, T, BinaryOperation>(
      Identity, WGSize, NWItems, usm::alloc::shared);
  test<KernelNameGroup<Name, class HostCase>, false, T, BinaryOperation>(
      Identity, WGSize, NWItems, usm::alloc::host);
  test<KernelNameGroup<Name, class DeviceCase>, false, T, BinaryOperation>(
      Identity, WGSize, NWItems, usm::alloc::device);
#endif
}

int main() {
  // fast atomics and fast reduce
  testUSM<class AtomicReduce1, int, ONEAPI::plus<int>>(0, 49, 49);
  testUSM<class AtomicReduce2, int, ONEAPI::plus<int>>(0, 8, 32);

  // fast atomics
  testUSM<class Atomic1, int, ONEAPI::bit_or<int>>(0, 7, 7 * 3);
  testUSM<class Atomic2, int, ONEAPI::bit_or<int>>(0, 4, 32);

  // fast reduce
  testUSM<class Reduce1, float, ONEAPI::minimum<float>>(
      getMaximumFPValue<float>(), 17, 17);
  testUSM<class Reduce2, float, ONEAPI::maximum<float>>(
      getMinimumFPValue<float>(), 4, 32);

  // generic algorithm
  testUSM<class Generic1, int, std::multiplies<int>>(1, 7, 7);
  testUSM<class Generic2, CustomVec<short>, CustomVecPlus<short>>(
      CustomVec<short>(0), 8, 8 * 3);

  std::cout << "Test passed\n";
  return 0;
}
