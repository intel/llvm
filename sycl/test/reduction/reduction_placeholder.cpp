// UNSUPPORTED: cuda
// Reductions use work-group builtins not yet supported by CUDA.

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// RUNx: env SYCL_DEVICE_TYPE=HOST %t.out
// TODO: Enable the test for HOST when it supports intel::reduce() and barrier()

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// with reductions initialized with a placeholder accessor.

#include "reduction_utils.hpp"
#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

template <typename T, int Dim, class BinaryOperation>
class SomeClass;

template <typename T, int Dim, class BinaryOperation>
void test(T Identity, size_t WGSize, size_t NWItems) {
  // Initialize.
  T CorrectOut;
  BinaryOperation BOp;

  buffer<T, 1> OutBuf(1);
  buffer<T, 1> InBuf(NWItems);
  initInputData(InBuf, CorrectOut, Identity, BOp, NWItems);

  (OutBuf.template get_access<access::mode::write>())[0] = Identity;

  auto Out = accessor<T, Dim, access::mode::read_write,
                      access::target::global_buffer,
                      access::placeholder::true_t>(OutBuf);
  // Compute.
  queue Q;
  Q.submit([&](handler &CGH) {
    auto In = InBuf.template get_access<access::mode::read>(CGH);
    CGH.require(Out);
    auto Redu = intel::reduction(Out, Identity, BinaryOperation());
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
  T ReduVar = (OutBuf.template get_access<access::mode::read>())[0];
  if (ReduVar != CorrectOut) {
    std::cout << "NWItems = " << NWItems << ", WGSize = " << WGSize << "\n";
    std::cout << "Computed value: " << ReduVar
              << ", Expected value: " << CorrectOut << "\n";
    assert(0 && "Wrong value.");
  }
}

int main() {
  // fast atomics and fast reduce
  test<int, 1, intel::plus<int>>(0, 49, 49 * 5);
  test<int, 0, intel::plus<int>>(0, 8, 8);

  // fast atomics
  test<int, 0, intel::bit_or<int>>(0, 7, 7 * 3);
  test<int, 1, intel::bit_or<int>>(0, 4, 128);

  // fast reduce
  test<float, 1, intel::minimum<float>>(getMaximumFPValue<float>(), 5, 5 * 7);
  test<float, 0, intel::maximum<float>>(getMinimumFPValue<float>(), 4, 128);

  // generic algorithm
  test<int, 0, std::multiplies<int>>(1, 7, 7 * 5);
  test<int, 1, std::multiplies<int>>(1, 8, 16);
  test<CustomVec<short>, 0, CustomVecPlus<short>>(CustomVec<short>(0), 8, 8 * 3);

  std::cout << "Test passed\n";
  return 0;
}
