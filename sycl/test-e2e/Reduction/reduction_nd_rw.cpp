// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// `Group algorithms are not supported on host device.` on Nvidia.
// XFAIL: hip_nvidia

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// with reductions initialized with a one element buffer.

#include "reduction_utils.hpp"

using namespace sycl;

int NumErrors = 0;

template <typename Name, typename T, class BinaryOperation>
void tests(queue &Q, T Identity, T Init, BinaryOperation BOp, size_t WGSize,
           size_t NWItems) {
  nd_range<1> NDRange(range<1>{NWItems}, range<1>{WGSize});
  NumErrors += test<Name>(Q, Identity, Init, BOp, NDRange);
}

template <typename Name, typename T, class BinaryOperation>
void tests(queue &Q, T Init, BinaryOperation BOp, size_t WGSize,
           size_t NWItems) {
  nd_range<1> NDRange(range<1>{NWItems}, range<1>{WGSize});
  NumErrors += test<Name>(Q, Init, BOp, NDRange);
}

int main() {
  queue Q;
  printDeviceInfo(Q);

  // Check non power-of-two work-group sizes.
  tests<class A1, int>(Q, 0, 99, std::plus<int>{}, 1, 7);
  tests<class A2, int>(Q, 0, -99, std::plus<int>{}, 49, 49 * 5);

  // Try some power-of-two work-group sizes.
  tests<class B1, int>(Q, 0, 99, std::plus<>{}, 2, 32);
  tests<class B2, int>(Q, 0, 199, std::plus<>{}, 32, 32);
  tests<class B3, int>(Q, 0, 299, std::plus<>{}, 128, 256);
  tests<class B4, int>(Q, 0, 399, std::plus<>{}, 256, 256);

  // Check with various operations and types.
  tests<class C1, int>(Q, 0, 99, std::plus<>{}, 1, 32);
  tests<class C2, int>(Q, 0, 99, std::bit_or<>{}, 8, 128);
  tests<class C3, int>(Q, 0, 99, std::bit_xor<>{}, 16, 256);
  tests<class C4, int>(Q, ~0, 99, std::bit_and<>{}, 32, 256);
  tests<class C5, int>(Q, (std::numeric_limits<int>::max)(), -99,
                       ext::oneapi::minimum<>{}, 64, 256);
  tests<class C6, int>(Q, (std::numeric_limits<int>::min)(), 99,
                       ext::oneapi::maximum<>{}, 128, 256);
  tests<class C7, int>(Q, 0, 99, std::plus<>{}, 256, 256);
  tests<class C8, int>(Q, 1, 2, std::multiplies<>{}, 8, 8);
  tests<class C9, float>(Q, 1, 1.2, std::multiplies<>{}, 8, 16);

  // Check with CUSTOM type.
  using CV = CustomVec<long long>;
  tests<class D1, CV>(Q, CV(0), CV(-199), CustomVecPlus<long long>{}, 8, 256);
  tests<class D2, CV>(Q, CV(-199), CustomVecPlus<long long>{}, 8, 256);

  // Check non power-of-two work-group sizes without identity.
  tests<class E1, int>(Q, 99, PlusWithoutIdentity<int>{}, 1, 7);
  tests<class E2, int>(Q, -99, PlusWithoutIdentity<int>{}, 49, 49 * 5);

  // Try some power-of-two work-group sizes without identity.
  tests<class F1, int>(Q, 99, PlusWithoutIdentity<int>{}, 2, 32);
  tests<class F2, int>(Q, 199, PlusWithoutIdentity<int>{}, 32, 32);
  tests<class F3, int>(Q, 299, PlusWithoutIdentity<int>{}, 128, 256);
  tests<class F4, int>(Q, 399, PlusWithoutIdentity<int>{}, 256, 256);

  printFinalStatus(NumErrors);
  return NumErrors;
}
