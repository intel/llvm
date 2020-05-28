// UNSUPPORTED: cuda
// Reductions use work-group builtins not yet supported by CUDA.

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// RUNx: env SYCL_DEVICE_TYPE=HOST %t.out
// TODO: Enable the test for HOST when it supports intel::reduce() and barrier()

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// with reductions initialized with USM var.

#include "reduction_utils.hpp"
#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

template <typename T, int Dim, class BinaryOperation>
class SomeClass;
template <typename T, int Dim, class BinaryOperation>
class Copy1;

template <typename T, int Dim, class BinaryOperation>
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
    event E = Q.submit([&](handler &CGH) {
      CGH.single_task<class Copy1<T, Dim, BinaryOperation>>([=]() {
        *ReduVarPtr = Identity;
      });
    });
    E.wait();
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
    auto Redu = intel::reduction(ReduVarPtr, Identity, BOp);
    range<1> GlobalRange(NWItems);
    range<1> LocalRange(WGSize);
    nd_range<1> NDRange(GlobalRange, LocalRange);
    CGH.parallel_for<SomeClass<T, Dim, BinaryOperation>>(
        NDRange, Redu, [=](nd_item<1> NDIt, auto &Sum) {
          Sum.combine(In[NDIt.get_global_linear_id()]);
        });
  });
  Q.wait();

  // Check correctness.
  T ComputedOut;
  if (AllocType == usm::alloc::device) {
    buffer<T, 1> Buf(&ComputedOut, range<1>(1));
    event E = Q.submit([&](handler &CGH) {
      auto OutAcc = Buf.template get_access<access::mode::discard_write>(CGH);
      CGH.copy(ReduVarPtr, OutAcc);
    });
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

template <typename T, int Dim, class BinaryOperation>
void testUSM(T Identity, size_t WGSize, size_t NWItems) {
  test<T, Dim, BinaryOperation>(Identity, WGSize, NWItems, usm::alloc::shared);
  test<T, Dim, BinaryOperation>(Identity, WGSize, NWItems, usm::alloc::host);
  test<T, Dim, BinaryOperation>(Identity, WGSize, NWItems, usm::alloc::device);
}

int main() {
  // fast atomics and fast reduce
  testUSM<int, 1, intel::plus<int>>(0, 49, 49 * 5);
  testUSM<int, 0, intel::plus<int>>(0, 8, 128);

  // fast atomics
  testUSM<int, 0, intel::bit_or<int>>(0, 7, 7 * 3);
  testUSM<int, 1, intel::bit_or<int>>(0, 4, 128);

  // fast reduce
  testUSM<float, 1, intel::minimum<float>>(getMaximumFPValue<float>(), 5, 5 * 7);
  testUSM<float, 0, intel::maximum<float>>(getMinimumFPValue<float>(), 4, 128);

  // generic algorithm
  testUSM<int, 0, std::multiplies<int>>(1, 7, 7 * 5);
  testUSM<int, 1, std::multiplies<int>>(1, 8, 16);
  testUSM<CustomVec<short>, 0, CustomVecPlus<short>>(
      CustomVec<short>(0), 8, 8 * 3);

  std::cout << "Test passed\n";
  return 0;
}
