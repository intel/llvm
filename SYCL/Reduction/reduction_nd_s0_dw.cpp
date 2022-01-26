// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// `Group algorithms are not supported on host device.` on HIP Nvidia.
// XFAIL: hip_nvidia

// TODO: test disabled due to sporadic fails in level_zero:gpu RT.
// UNSUPPORTED: linux && level_zero

// This test performs basic checks of parallel_for(nd_range, reduction, func)
// with reductions initialized with 0-dimensional discard_write accessor.

#include "reduction_nd_range_scalar.hpp"

using namespace cl::sycl;

int NumErrors = 0;

template <typename Name, typename T, class BinaryOperation>
void tests(queue &Q, T Identity, T Init, BinaryOperation BOp, size_t WGSize,
           size_t NWItems) {
  constexpr access::mode DW = access::mode::discard_write;
  nd_range<1> NDRange(range<1>{NWItems}, range<1>{WGSize});
  NumErrors +=
      test<Name, false /*SYCL2020*/, DW, 0>(Q, Identity, Init, BOp, NDRange);
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
  tests<class B4, int>(Q, 0, 99, std::bit_xor<>{}, 16, 256);
  tests<class B5, int>(Q, ~0, 99, std::bit_and<>{}, 32, 256);
  tests<class B6, int>(Q, (std::numeric_limits<int>::max)(), -99,
                       ext::oneapi::minimum<>{}, 64, 256);
  tests<class B7, int>(Q, (std::numeric_limits<int>::min)(), 99,
                       ext::oneapi::maximum<>{}, 128, 256);
  tests<class B8, int>(Q, 0, 99, std::plus<>{}, 256, 256);

  // Check with various types.
  tests<class C1, float>(Q, 1, 99, std::multiplies<>{}, 8, 16);
  tests<class C2, short>(Q, 0x7fff, -99, ext::oneapi::minimum<>{}, 8, 256);
  tests<class C3, unsigned char>(Q, 0, 99, ext::oneapi::maximum<>{}, 8, 256);

  // Check with CUSTOM type.
  using CV = CustomVec<long long>;
  tests<class D1, CV>(Q, CV(0), CV(99), CustomVecPlus<long long>{}, 8, 256);

  printFinalStatus(NumErrors);
  return NumErrors;
}
