// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// with reductions initialized with 0-dimensional read_write accessor.

// This test fails with exceeded time out on Windows with OpenCL, temporarily
// disabling
// UNSUPPORTED: windows && opencl

#include "reduction_utils.hpp"
#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

// This allocator is needed only for the purpose of testing buffers
// with allocator that is not same_as sycl::buffer_allocator.
struct CustomAllocator : public sycl::buffer_allocator {};

template <typename T, bool B> class KName;

template <typename Name, bool IsSYCL2020Mode, typename T, int Dim,
          class BinaryOperation>
void test(T Identity, size_t WGSize, size_t NWItems) {
  buffer<T, 1> InBuf(NWItems);
  buffer<T, 1, CustomAllocator> OutBuf(1);

  // Initialize.
  BinaryOperation BOp;
  T CorrectOut;
  initInputData(InBuf, CorrectOut, Identity, BOp, NWItems);

  (OutBuf.template get_access<access::mode::write>())[0] = Identity;

  // Compute.
  queue Q;
  nd_range<1> NDRange(range<1>{NWItems}, range<1>{WGSize});
  if constexpr (IsSYCL2020Mode) {
    Q.submit([&](handler &CGH) {
      auto In = InBuf.template get_access<access::mode::read>(CGH);
      auto Redu = sycl::reduction(OutBuf, CGH, Identity, BOp);

      CGH.parallel_for<Name>(NDRange, Redu, [=](nd_item<1> NDIt, auto &Sum) {
        Sum.combine(In[NDIt.get_global_linear_id()]);
      });
    });
  } else {
    Q.submit([&](handler &CGH) {
      auto In = InBuf.template get_access<access::mode::read>(CGH);
      accessor<T, Dim, access::mode::read_write, access::target::global_buffer>
          Out(OutBuf, CGH);
      auto Redu = ONEAPI::reduction(Out, Identity, BOp);

      range<1> GlobalRange(NWItems);
      range<1> LocalRange(WGSize);
      nd_range<1> NDRange(GlobalRange, LocalRange);
      CGH.parallel_for<Name>(NDRange, Redu, [=](nd_item<1> NDIt, auto &Sum) {
        Sum.combine(In[NDIt.get_global_linear_id()]);
      });
    });
  }

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

template <typename Name, typename T, int Dim, class BinaryOperation>
void testBoth(T Identity, size_t WGSize, size_t NWItems) {
  test<KName<Name, false>, false, T, Dim, BinaryOperation>(Identity, WGSize,
                                                           NWItems);
  test<KName<Name, true>, true, T, Dim, BinaryOperation>(Identity, WGSize,
                                                         NWItems);
}

int main() {
  // Check some less standards WG sizes and corner cases first.
  testBoth<class A, int, 0, ONEAPI::plus<int>>(0, 2, 2);
  testBoth<class B, int, 0, ONEAPI::plus<int>>(0, 7, 7);
  testBoth<class C, int, 0, ONEAPI::plus<int>>(0, 9, 18);
  testBoth<class D, int, 0, ONEAPI::plus<int>>(0, 49, 49 * 5);

  // Try some power-of-two work-group sizes.
  testBoth<class E, int, 0, ONEAPI::plus<int>>(0, 2, 64);
  testBoth<class F, int, 0, ONEAPI::plus<int>>(0, 4, 64);
  testBoth<class G, int, 0, ONEAPI::plus<int>>(0, 8, 128);
  testBoth<class H, int, 0, ONEAPI::plus<int>>(0, 16, 256);
  testBoth<class I, int, 0, ONEAPI::plus<int>>(0, 32, 256);
  testBoth<class J, int, 0, ONEAPI::plus<int>>(0, 64, 256);
  testBoth<class K, int, 0, ONEAPI::plus<int>>(0, 128, 256);
  testBoth<class L, int, 0, ONEAPI::plus<int>>(0, 256, 256);

  // Check with various operations.
  testBoth<class M, int, 0, std::multiplies<int>>(1, 8, 256);
  testBoth<class N, int, 0, ONEAPI::bit_or<int>>(0, 8, 256);
  testBoth<class O, int, 0, ONEAPI::bit_xor<int>>(0, 8, 256);
  testBoth<class P, int, 0, ONEAPI::bit_and<int>>(~0, 8, 256);
  testBoth<class Q, int, 0, ONEAPI::minimum<int>>(
      (std::numeric_limits<int>::max)(), 8, 256);
  testBoth<class R, int, 0, ONEAPI::maximum<int>>(
      (std::numeric_limits<int>::min)(), 8, 256);

  // Check with various types.
  testBoth<class S, float, 0, std::multiplies<float>>(1, 8, 256);
  testBoth<class T, float, 0, ONEAPI::minimum<float>>(
      getMaximumFPValue<float>(), 8, 256);
  testBoth<class U, float, 0, ONEAPI::maximum<float>>(
      getMinimumFPValue<float>(), 8, 256);

  // Check with CUSTOM type.
  testBoth<class V, CustomVec<long long>, 0, CustomVecPlus<long long>>(
      CustomVec<long long>(0), 8, 256);

  std::cout << "Test passed\n";
  return 0;
}
