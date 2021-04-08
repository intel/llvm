// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// with reductions initialized with 0-dimensional discard_write accessor.

#include "reduction_utils.hpp"
#include <CL/sycl.hpp>
#include <cassert>

using namespace cl::sycl;

// This allocator is needed only for the purpose of testing buffers
// with allocator that is not same_as sycl::buffer_allocator.
struct CustomAllocator : public sycl::buffer_allocator {};

template <typename T, bool B> class KName;

template <typename Name, bool IsSYCL2020Mode, typename T, class BinaryOperation>
void test(queue &Q, T Identity, T Init, size_t WGSize, size_t NWItems) {
  buffer<T, 1> InBuf(NWItems);
  buffer<T, 1, CustomAllocator> OutBuf(1);

  // Initialize.
  BinaryOperation BOp;
  T CorrectOut;
  initInputData(InBuf, CorrectOut, Identity, BOp, NWItems);

  // This assignment is discarded, but is still needed to verify
  // that is discarded.
  (OutBuf.template get_access<access::mode::write>())[0] = Init;

  // Compute.

  nd_range<1> NDRange(range<1>{NWItems}, range<1>{WGSize});
  if constexpr (IsSYCL2020Mode) {
    Q.submit([&](handler &CGH) {
      auto In = InBuf.template get_access<access::mode::read>(CGH);
      property_list PropList(property::reduction::initialize_to_identity{});
      auto Redu = sycl::reduction(OutBuf, CGH, Identity, BOp, PropList);

      CGH.parallel_for<Name>(NDRange, Redu, [=](nd_item<1> NDIt, auto &Sum) {
        Sum.combine(In[NDIt.get_global_linear_id()]);
      });
    });
  } else {
    Q.submit([&](handler &CGH) {
      auto In = InBuf.template get_access<access::mode::read>(CGH);
      accessor<T, 0, access::mode::discard_write, access::target::global_buffer>
          Out(OutBuf, CGH);
      auto Redu = ONEAPI::reduction(Out, Identity, BOp);

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

template <typename Name, typename T, class BinaryOperation>
void testBoth(queue &Q, T Identity, T Init, size_t WGSize, size_t NWItems) {
  test<KName<Name, false>, false, T, BinaryOperation>(Q, Identity, Init, WGSize,
                                                      NWItems);
  test<KName<Name, true>, true, T, BinaryOperation>(Q, Identity, Init, WGSize,
                                                    NWItems);
}

int main() {
  queue Q;

  // Check some non power-of-two work-group sizes.
  testBoth<class A1, int, ONEAPI::plus<int>>(Q, 0, 99, 1, 7);
  testBoth<class A2, int, ONEAPI::plus<int>>(Q, 0, 99, 49, 49 * 5);

  // Try some power-of-two work-group sizes.
  testBoth<class B1, int, ONEAPI::plus<int>>(Q, 0, 99, 1, 32);
  testBoth<class B2, int, std::multiplies<int>>(Q, 1, 99, 4, 32);
  testBoth<class B4, int, ONEAPI::bit_xor<int>>(Q, 0, 99, 16, 256);
  testBoth<class B5, int, ONEAPI::bit_and<int>>(Q, ~0, 99, 32, 256);
  testBoth<class B6, int, ONEAPI::minimum<int>>(
      Q, (std::numeric_limits<int>::max)(), -99, 64, 256);
  testBoth<class B7, int, ONEAPI::maximum<int>>(
      Q, (std::numeric_limits<int>::min)(), 99, 128, 256);
  testBoth<class B8, int, ONEAPI::plus<>>(Q, 0, 99, 256, 256);

  // Check with various types.
  testBoth<class C1, float, std::multiplies<>>(Q, 1, 99, 8, 256);
  testBoth<class C2, short, ONEAPI::minimum<>>(Q, 0x7fff, -99, 8, 256);
  testBoth<class C3, unsigned char, ONEAPI::maximum<>>(Q, 0, 99, 8, 256);

  // Check with CUSTOM type.
  testBoth<class D1, CustomVec<long long>, CustomVecPlus<long long>>(
      Q, CustomVec<long long>(0), CustomVec<long long>(99), 8, 256);

  std::cout << "Test passed\n";
  return 0;
}
