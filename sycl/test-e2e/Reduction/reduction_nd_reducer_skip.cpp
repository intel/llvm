// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// Group algorithms are not supported on Nvidia.
// XFAIL: hip_nvidia

// Windows doesn't yet have full shutdown().
// UNSUPPORTED: ze_debug && windows

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// with reductions initialized with a one element buffer. Additionally, some
// reducers will not be written to.

#include "reduction_utils.hpp"

using namespace sycl;

int NumErrors = 0;

template <typename T> class SkipEvenName;
template <typename T> class SkipOddName;
template <typename T> class SkipAllName;

template <typename Name, typename T, class BinaryOperation>
void tests(queue &Q, T Identity, T Init, BinaryOperation BOp, size_t WGSize,
           size_t NWItems) {
  nd_range<1> NDRange(range<1>{NWItems}, range<1>{WGSize});
  NumErrors += test<SkipEvenName<Name>, T>(Q, Identity, Init, BOp, NDRange,
                                           property_list{}, SkipEvenOp{});
  NumErrors += test<SkipOddName<Name>, T>(Q, Identity, Init, BOp, NDRange,
                                          property_list{}, SkipOddOp{});
  NumErrors += test<SkipAllName<Name>, T>(Q, Identity, Init, BOp, NDRange,
                                          property_list{}, SkipAllOp{});
}

int main() {
  queue Q;
  printDeviceInfo(Q);

  // Check some non power-of-two work-group sizes.
  tests<class A1, int>(Q, 0, 99, std::plus<int>{}, 1, 7);
  tests<class A2, int>(Q, 0, 99, std::plus<int>{}, 49, 49 * 5);

  // Try some power-of-two work-group sizes.
  tests<class B1, int>(Q, 0, 99, std::plus<>{}, 1, 32);
  tests<class B2, int>(Q, 1, 99, std::multiplies<>{}, 4, 32);
  tests<class B3, int>(Q, 0, 99, std::bit_or<>{}, 8, 128);
  tests<class B4, int>(Q, 0, 99, std::bit_xor<>{}, 16, 256);
  tests<class B5, int>(Q, ~0, 99, std::bit_and<>{}, 32, 256);
  tests<class B6, int>(Q, (std::numeric_limits<int>::max)(), -99,
                       ext::oneapi::minimum<>{}, 64, 256);
  tests<class B7, int>(Q, (std::numeric_limits<int>::min)(), 99,
                       ext::oneapi::maximum<>{}, 128, 256);
  tests<class B8, int>(Q, 0, 99, std::plus<>{}, 256, 256);

  // Check with various types.
  tests<class C1, float>(Q, 1, 99, std::multiplies<>{}, 8, 24);
  tests<class C2, short>(Q, 0x7fff, -99, ext::oneapi::minimum<>{}, 8, 256);
  tests<class C3, unsigned char>(Q, 0, 99, ext::oneapi::maximum<>{}, 8, 256);

  // Check with CUSTOM type.
  using CV = CustomVec<long long>;
  tests<class D1, CV>(Q, CV(0), CV(99), CustomVecPlus<long long>{}, 8, 256);

  printFinalStatus(NumErrors);
  return NumErrors;
}
