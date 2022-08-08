// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// `Group algorithms are not supported on host device.` on Nvidia.
// XFAIL: hip_nvidia

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// with reductions initialized with 1-dimensional read_write accessor
// accessing 1 element buffer.

#include "reduction_nd_range_scalar.hpp"

using namespace sycl;

int NumErrors = 0;

template <typename Name, typename T, class BinaryOperation>
void tests(queue &Q, T Identity, T Init, BinaryOperation BOp, size_t WGSize,
           size_t NWItems) {
  constexpr access::mode RW = access::mode::read_write;
  nd_range<1> NDRange(range<1>{NWItems}, range<1>{WGSize});
  NumErrors += testBoth<Name, RW>(Q, Identity, Init, BOp, NDRange);
}

int main() {
  queue Q;
  printDeviceInfo(Q);

  // Check non power-of-two work-group sizes.
  tests<class A1, int>(Q, 0, 99, std::plus<int>{}, 1, 7);
  tests<class A2, int>(Q, 0, -99, std::plus<int>{}, 49, 49 * 5);

  // Check with various operations and types.
  tests<class B1, int>(Q, 0, 99, std::plus<>{}, 1, 32);
  tests<class B2, int>(Q, 0, 99, std::bit_or<>{}, 8, 128);
  tests<class B3, int>(Q, 0, 99, std::bit_xor<>{}, 16, 256);
  tests<class B4, int>(Q, ~0, 99, std::bit_and<>{}, 32, 256);
  tests<class B5, int>(Q, (std::numeric_limits<int>::max)(), -99,
                       ext::oneapi::minimum<>{}, 64, 256);
  tests<class B6, int>(Q, (std::numeric_limits<int>::min)(), 99,
                       ext::oneapi::maximum<>{}, 128, 256);
  tests<class B7, int>(Q, 0, 99, std::plus<>{}, 256, 256);

  // Check with CUSTOM type.
  using CV = CustomVec<long long>;
  tests<class C1, CV>(Q, CV(0), CV(-199), CustomVecPlus<long long>{}, 8, 256);

  printFinalStatus(NumErrors);
  return NumErrors;
}
