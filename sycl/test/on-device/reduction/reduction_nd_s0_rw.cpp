// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUNx: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// with reductions initialized with 0-dimensional read_write accessor.

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

  (OutBuf.template get_access<access::mode::write>())[0] = Identity;

  // Compute.
  queue Q;
  Q.submit([&](handler &CGH) {
    auto In = InBuf.template get_access<access::mode::read>(CGH);
    accessor<T, Dim, access::mode::read_write, access::target::global_buffer>
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
  test<class KernelName_oXfFQUctpKif, int, 0, ONEAPI::plus<int>>(0, 2, 2);
  test<class KernelName_YnoUqtntrJ, int, 0, ONEAPI::plus<int>>(0, 7, 7);
  test<class KernelName_hfCsGycSpg, int, 0, ONEAPI::plus<int>>(0, 9, 18);
  test<class KernelName_OtxLfwQuVfGAUEXMIs, int, 0, ONEAPI::plus<int>>(0, 49,
                                                                       49 * 5);

  // Try some power-of-two work-group sizes.
  test<class KernelName_lMJpe, int, 0, ONEAPI::plus<int>>(0, 2, 64);
  test<class KernelName_jikSUrEuFUxYGGfXNet, int, 0, ONEAPI::plus<int>>(0, 4,
                                                                        64);
  test<class KernelName_cByxQmddzgEGUeboDDbO, int, 0, ONEAPI::plus<int>>(0, 8,
                                                                         128);
  test<class KernelName_pggyS, int, 0, ONEAPI::plus<int>>(0, 16, 256);
  test<class KernelName_CWZouFJ, int, 0, ONEAPI::plus<int>>(0, 32, 256);
  test<class KernelName_IjuYfJxWZdaVMdE, int, 0, ONEAPI::plus<int>>(0, 64, 256);
  test<class KernelName_tcKhlzfhg, int, 0, ONEAPI::plus<int>>(0, 128, 256);
  test<class KernelName_eWffIBPdwvvUwPFZFeG, int, 0, ONEAPI::plus<int>>(0, 256,
                                                                        256);

  // Check with various operations.
  test<class KernelName_rWAaJsLUS, int, 0, std::multiplies<int>>(1, 8, 256);
  test<class KernelName_jZoWyBoLxybjrbk, int, 0, ONEAPI::bit_or<int>>(0, 8,
                                                                      256);
  test<class KernelName_jdixaAPjypPSGPCbXIw, int, 0, ONEAPI::bit_xor<int>>(0, 8,
                                                                           256);
  test<class KernelName_FNGt, int, 0, ONEAPI::bit_and<int>>(~0, 8, 256);
  test<class KernelName_KPtKKagKhZzwSibEl, int, 0, ONEAPI::minimum<int>>(
      (std::numeric_limits<int>::max)(), 8, 256);
  test<class KernelName_xdNhx, int, 0, ONEAPI::maximum<int>>(
      (std::numeric_limits<int>::min)(), 8, 256);

  // Check with various types.
  test<class KernelName_IxDwu, float, 0, std::multiplies<float>>(1, 8, 256);
  test<class KernelName_NpYzX, float, 0, ONEAPI::minimum<float>>(
      getMaximumFPValue<float>(), 8, 256);
  test<class KernelName_dofjVNlXWgJ, float, 0, ONEAPI::maximum<float>>(
      getMinimumFPValue<float>(), 8, 256);

  // Check with CUSTOM type.
  test<class KernelName_XrOnrVnB, CustomVec<long long>, 0,
       CustomVecPlus<long long>>(CustomVec<long long>(0), 8, 256);

  std::cout << "Test passed\n";
  return 0;
}
