// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// with reductions initialized with USM var. It tests only SYCL-2020 reduction
// (sycl::reduction) assuming discard-write access, i.e. when reduction
// is created with property::reduction::initialize_to_identity.

#include "reduction_utils.hpp"
#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

template <typename T1, typename T2> class KernelNameGroup;

template <typename Name, typename T, class BinaryOperation>
void test(queue &Q, T Identity, T Init, size_t WGSize, size_t NWItems,
          usm::alloc AllocType) {
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
       CGH.single_task<KernelNameGroup<Name, class InitKernel>>(
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
     property_list PropList(property::reduction::initialize_to_identity{});
     auto Redu = sycl::reduction(ReduVarPtr, Identity, BOp, PropList);
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
    std::cerr << "NWItems = " << NWItems << ", WGSize = " << WGSize << "\n";
    std::cerr << "Computed value: " << ComputedOut
              << ", Expected value: " << CorrectOut
              << ", AllocMode: " << static_cast<int>(AllocType) << "\n";
    assert(0 && "Wrong value.");
  }

  free(ReduVarPtr, Q.get_context());
}

template <typename Name, typename T, class BinaryOperation>
void testUSM(queue &Q, T Identity, T Init, size_t WGSize, size_t NWItems) {
  test<KernelNameGroup<Name, class Shared2020>, T, BinaryOperation>(
      Q, Identity, Init, WGSize, NWItems, usm::alloc::shared);
  test<KernelNameGroup<Name, class Host2020>, T, BinaryOperation>(
      Q, Identity, Init, WGSize, NWItems, usm::alloc::host);
  test<KernelNameGroup<Name, class Device2020>, T, BinaryOperation>(
      Q, Identity, Init, WGSize, NWItems, usm::alloc::device);
}

int main() {
  queue Q;
  // fast atomics and fast reduce
  testUSM<class AtomicReduce1, int, ONEAPI::plus<int>>(Q, 0, 99, 49, 5 * 49);

  // fast atomics
  testUSM<class Atomic1, int, ONEAPI::bit_or<int>>(Q, 0, 0xff00ff00, 7, 7);
  testUSM<class Atomic2, int, ONEAPI::bit_or<int>>(Q, 0, 0x7f007f00, 4, 32);

  // fast reduce
  testUSM<class Reduce1, float, ONEAPI::minimum<float>>(
      Q, getMaximumFPValue<float>(), -100.0, 17, 17);
  testUSM<class Reduce2, float, ONEAPI::maximum<float>>(
      Q, getMinimumFPValue<float>(), 100.0, 4, 32);

  // generic algorithm
  testUSM<class Generic1, int, std::multiplies<int>>(Q, 1, 5, 7, 7);
  testUSM<class Generic2, CustomVec<short>, CustomVecPlus<short>>(
      Q, CustomVec<short>(0), CustomVec<short>(77), 8, 8 * 3);

  std::cout << "Test passed\n";
  return 0;
}
