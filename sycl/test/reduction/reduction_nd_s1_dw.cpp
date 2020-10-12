// UNSUPPORTED: cuda
// Reductions use work-group builtins not yet supported by CUDA.
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUNx: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// with reductions initialized with 1-dimensional discard_write accessor
// accessing 1 element buffer.

#include "reduction_utils.hpp"
#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

template <typename SpecializationKernelName, typename T, int Dim,
          class BinaryOperation>
void test(T Identity, size_t WGSize, size_t NWItems) {
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
    auto Redu = ONEAPI::reduction(Out, Identity, BOp);

    range<1> GlobalRange(NWItems);
    range<1> LocalRange(WGSize);
    nd_range<1> NDRange(GlobalRange, LocalRange);
    CGH.parallel_for<SpecializationKernelName>(
        NDRange, Redu, [=](nd_item<1> NDIt, auto &Sum) {
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
  // Check some less standards WG sizes and corner cases first.
  test<class KernelName_YvZmtZPNMqkXPfNQEqtU, int, 1, ONEAPI::plus<int>>(0, 2,
                                                                         2);
  test<class KernelName_jxkTUxpuBScJxMzUuVcO, int, 1, ONEAPI::plus<int>>(0, 7,
                                                                         7);
  test<class KernelName_ySnvYMbreHXDCwJSbSaH, int, 1, ONEAPI::plus<int>>(0, 9,
                                                                         18);
  test<class KernelName_UjxromGtR, int, 1, ONEAPI::plus<int>>(0, 49, 49 * 5);

  // Try some power-of-two work-group sizes.
  test<class KernelName_EnupfpikIEjljnjxsR, int, 1, ONEAPI::plus<int>>(0, 2,
                                                                       64);
  test<class KernelName_rjRdxjtMBBG, int, 1, ONEAPI::plus<int>>(0, 4, 64);
  test<class KernelName_EpopkaQNRW, int, 1, ONEAPI::plus<int>>(0, 8, 128);
  test<class KernelName_OcP, int, 1, ONEAPI::plus<int>>(0, 16, 256);
  test<class KernelName_lewnPBkCvTXJYXfNQs, int, 1, ONEAPI::plus<int>>(0, 32,
                                                                       256);
  test<class KernelName_sEtIX, int, 1, ONEAPI::plus<int>>(0, 64, 256);
  test<class KernelName_yyXXOSx, int, 1, ONEAPI::plus<int>>(0, 128, 256);
  test<class KernelName_BebxPvIBvukTydfg, int, 1, ONEAPI::plus<int>>(0, 256,
                                                                     256);

  // Check with various operations.
  test<class KernelName_LaYCRJkYwJUzM, int, 1, std::multiplies<int>>(1, 8, 256);
  test<class KernelName_rRbqrnaCoinW, int, 1, ONEAPI::bit_or<int>>(0, 8, 256);
  test<class KernelName_WSadjhrpTq, int, 1, ONEAPI::bit_xor<int>>(0, 8, 256);
  test<class KernelName_OigvbFHsl, int, 1, ONEAPI::bit_and<int>>(~0, 8, 256);
  test<class KernelName_ZHcNbu, int, 1, ONEAPI::minimum<int>>(
      (std::numeric_limits<int>::max)(), 8, 256);
  test<class KernelName_JzbhkW, int, 1, ONEAPI::maximum<int>>(
      (std::numeric_limits<int>::min)(), 8, 256);

  // Check with various types.
  test<class KernelName_XeQbNSZBWhbf, float, 1, std::multiplies<float>>(1, 8,
                                                                        256);
  test<class KernelName_Otl, float, 1, ONEAPI::minimum<float>>(
      getMaximumFPValue<float>(), 8, 256);
  test<class KernelName_KhyYgqwIgLKroAzjr, float, 1, ONEAPI::maximum<float>>(
      getMinimumFPValue<float>(), 8, 256);

  // Check with CUSTOM type.
  test<class KernelName_VzSVAWkAmHq, CustomVec<long long>, 1,
       CustomVecPlus<long long>>(CustomVec<long long>(0), 8, 256);

  std::cout << "Test passed\n";
  return 0;
}
