// UNSUPPORTED: cuda
// Reductions use work-group builtins (e.g. intel::reduce()) not yet supported
// by CUDA.
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUNx: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test performs basic checks of parallel_for(nd_range, reduction, lambda)

#include "reduction_utils.hpp"
#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

template <class KernelName, typename T, class BinaryOperation>
void test(T Identity, BinaryOperation BOp, size_t WGSize, size_t NWItems) {
  buffer<T, 1> InBuf(NWItems);
  buffer<T, 1> OutBuf(1);

  // Initialize.
  T CorrectOut;
  initInputData(InBuf, CorrectOut, Identity, BOp, NWItems);

  // Compute.
  queue Q;
  Q.submit([&](handler &CGH) {
    auto In = InBuf.template get_access<access::mode::read>(CGH);
    auto Out = OutBuf.template get_access<access::mode::discard_write>(CGH);
    auto Redu = intel::reduction(Out, Identity, BOp);

    range<1> GlobalRange(NWItems);
    range<1> LocalRange(WGSize);
    nd_range<1> NDRange(GlobalRange, LocalRange);
    CGH.parallel_for<KernelName>(NDRange, Redu,
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

int main() {
  test<class AddTestName, int>(
      0, [](auto x, auto y) { return (x + y); }, 8, 32);
  test<class MulTestName, int>(
      0, [](auto x, auto y) { return (x * y); }, 8, 32);

  // Check with CUSTOM type.
  test<class CustomAddTestname, CustomVec<long long>>(
      CustomVec<long long>(0),
      [](auto x, auto y) {
        CustomVecPlus<long long> BOp;
        return BOp(x, y);
      },
      4, 64);

  std::cout << "Test passed\n";
  return 0;
}
