// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// RUNx: %HOST_RUN_PLACEHOLDER %t.out
// TODO: Enable the test for HOST when it supports ONEAPI::reduce() and
// barrier()

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// with reductions initialized with a placeholder accessor.

#include "reduction_utils.hpp"
#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

template <typename... Ts> class KNGroup;

template <typename SpecializationKernelName, typename T, int Dim,
          class BinaryOperation, access::mode Mode>
void testOneCase(T Identity, T Init, size_t WGSize, size_t NWItems) {
  // Initialize.
  T CorrectOut;
  BinaryOperation BOp;

  buffer<T, 1> OutBuf(1);
  buffer<T, 1> InBuf(NWItems);
  initInputData(InBuf, CorrectOut, Identity, BOp, NWItems);
  if (Mode == access::mode::read_write)
    CorrectOut = BOp(CorrectOut, Init);

  (OutBuf.template get_access<access::mode::write>())[0] = Init;

  auto Out = accessor<T, Dim, Mode, access::target::global_buffer,
                      access::placeholder::true_t>(OutBuf);
  // Compute.
  queue Q;
  Q.submit([&](handler &CGH) {
    auto In = InBuf.template get_access<access::mode::read>(CGH);
    CGH.require(Out);
    auto Redu = ONEAPI::reduction(Out, Identity, BinaryOperation());
    range<1> GlobalRange(NWItems);
    range<1> LocalRange(WGSize);
    nd_range<1> NDRange(GlobalRange, LocalRange);
    CGH.parallel_for<SpecializationKernelName>(
        NDRange, Redu, [=](nd_item<1> NDIt, auto &Sum) {
          Sum.combine(In[NDIt.get_global_linear_id()]);
        });
  });
  Q.wait();

  // Check correctness.
  T ReduVar = (OutBuf.template get_access<access::mode::read>())[0];
  if (ReduVar != CorrectOut) {
    std::cout << "NWItems = " << NWItems << ", WGSize = " << WGSize << "\n";
    std::cout << "Computed value: " << ReduVar
              << ", Expected value: " << CorrectOut << "\n";
    assert(0 && "Wrong value.");
  }
}

template <typename KernelName, typename T, int Dim, class BinaryOperation>
void test(T Identity, T Init, size_t WGSize, size_t NWItems) {
  testOneCase<KNGroup<KernelName, class RWCase>, T, Dim, BinaryOperation,
              access::mode::read_write>(Identity, Init, WGSize, NWItems);
  testOneCase<KNGroup<KernelName, class DWCase>, T, Dim, BinaryOperation,
              access::mode::discard_write>(Identity, Init, WGSize, NWItems);
}

int main() {
  // fast atomics and fast reduce
  test<class AtomicReduce1, int, 1, ONEAPI::plus<int>>(0, 77, 49, 49 * 5);
  test<class AtomicReduce2, int, 0, ONEAPI::plus<int>>(0, -77, 8, 8);

  // fast atomics
  test<class Atomic1, int, 0, ONEAPI::bit_or<int>>(0, 233, 7, 7 * 3);
  test<class Atomic2, int, 1, ONEAPI::bit_or<int>>(0, 177, 4, 128);

  // fast reduce
  test<class Reduce1, float, 1, ONEAPI::minimum<float>>(
      getMaximumFPValue<float>(), -5.0, 5, 5 * 7);
  test<class Reduce2, float, 0, ONEAPI::maximum<float>>(
      getMinimumFPValue<float>(), -5.0, 4, 128);

  // generic algorithm
  test<class Generic1, int, 0, std::multiplies<int>>(1, 2, 7, 7 * 5);
  test<class Generic2, int, 1, std::multiplies<int>>(1, 3, 8, 16);
  test<class Generic3, CustomVec<short>, 0, CustomVecPlus<short>>(
      CustomVec<short>(0), CustomVec<short>(4), 8, 8 * 3);

  std::cout << "Test passed\n";
  return 0;
}
