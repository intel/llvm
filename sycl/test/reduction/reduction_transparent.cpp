// UNSUPPORTED: cuda
// Reductions use work-group builtins not yet supported by CUDA.

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

// RUNx: env SYCL_DEVICE_TYPE=HOST %t.out
// TODO: Enable the test for HOST when it supports ONEAPI::reduce() and
// barrier()

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// where func is a transparent functor.

#include "reduction_utils.hpp"
#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

template <typename... Ts> class KernelNameGroup;

// Checks reductions initialized with transparent functor and explicitly set
// identity value.
template <typename SpecializationKernelName, typename T, int Dim,
          class BinaryOperation>
void testId(T Identity, size_t WGSize, size_t NWItems) {
  buffer<T, 1> InBuf(NWItems);
  buffer<T, 1> OutBuf(1);

  // Initialize.
  BinaryOperation BOp;
  T CorrectOut;
  initInputData(InBuf, CorrectOut, Identity, BOp, NWItems);

  // Compute.
  queue Q;
  Q.submit([&](handler &CGH) {
    auto In = InBuf.template get_access<access::mode::read>(CGH);
    accessor<T, Dim, access::mode::discard_write, access::target::global_buffer>
        Out(OutBuf, CGH);

    range<1> GlobalRange(NWItems);
    range<1> LocalRange(WGSize);
    nd_range<1> NDRange(GlobalRange, LocalRange);
    CGH.parallel_for<SpecializationKernelName>(
        NDRange, ONEAPI::reduction(Out, Identity, BOp),
        [=](nd_item<1> NDIt, auto &Sum) {
          Sum.combine(In[NDIt.get_global_linear_id()]);
        });
  });

  // Check correctness.
  auto Out = OutBuf.template get_access<access::mode::read>();
  T ComputedOut = *(Out.get_pointer());
  if (ComputedOut != CorrectOut) {
    std::cout << "NWItems = " << NWItems << ", WGSize = " << WGSize << "\n";
    std::cout << "Computed value: " << ComputedOut
              << ", Expected value: " << CorrectOut << "\n";
    assert(0 && "Wrong value.");
  }
}

// Checks reductions initialized with transparent functor and identity
// value not explicitly specified. The parameter 'Identity' is passed here
// only to pre-initialize input data correctly.
template <typename SpecializationKernelName, typename T, int Dim,
          class BinaryOperation>
void testNoId(T Identity, size_t WGSize, size_t NWItems) {
  buffer<T, 1> InBuf(NWItems);
  buffer<T, 1> OutBuf(1);

  // Initialize.
  BinaryOperation BOp;
  T CorrectOut;
  initInputData(InBuf, CorrectOut, Identity, BOp, NWItems);

  // Compute.
  queue Q;
  Q.submit([&](handler &CGH) {
    auto In = InBuf.template get_access<access::mode::read>(CGH);
    accessor<T, Dim, access::mode::discard_write, access::target::global_buffer>
        Out(OutBuf, CGH);

    range<1> GlobalRange(NWItems);
    range<1> LocalRange(WGSize);
    nd_range<1> NDRange(GlobalRange, LocalRange);
    CGH.parallel_for<SpecializationKernelName>(
        NDRange, ONEAPI::reduction(Out, BOp), [=](nd_item<1> NDIt, auto &Sum) {
          Sum.combine(In[NDIt.get_global_linear_id()]);
        });
  });

  // Check correctness.
  auto Out = OutBuf.template get_access<access::mode::read>();
  T ComputedOut = *(Out.get_pointer());
  if (ComputedOut != CorrectOut) {
    std::cout << "NWItems = " << NWItems << ", WGSize = " << WGSize << "\n";
    std::cout << "Computed value: " << ComputedOut
              << ", Expected value: " << CorrectOut << "\n";
    assert(0 && "Wrong value.");
  }
}

template <typename SpecializationKernelName, typename T, int Dim,
          class BinaryOperation>
void test(T Identity, size_t WGSize, size_t NWItems) {
  testId<KernelNameGroup<SpecializationKernelName,
                         class KernelName_ObjsWYkZuXCCtNW>,
         T, Dim, BinaryOperation>(Identity, WGSize, NWItems);
  testNoId<KernelNameGroup<SpecializationKernelName,
                           class KernelName_WFtswXpcLpzOBO>,
           T, Dim, BinaryOperation>(Identity, WGSize, NWItems);
}

int main() {
#if __cplusplus >= 201402L
  test<class KernelName_slumazIfW, float, 0, ONEAPI::maximum<>>(
      getMinimumFPValue<float>(), 7, 7 * 5);
  test<class KernelName_XtRLKzVaIuL, signed char, 0, ONEAPI::plus<>>(0, 7, 49);
  test<class KernelName_adpasoZLtoLyZcczwrkV, unsigned char, 1,
       std::multiplies<>>(1, 4, 16);
  test<class KernelName_BZDXCHzCBhBb, unsigned short, 0, ONEAPI::plus<>>(
      0, 1, 512 + 32);
#endif // __cplusplus >= 201402L

  std::cout << "Test passed\n";
  return 0;
}
