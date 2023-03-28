// RUN: %clangxx -DENABLE_64_BIT=false -fsycl -fsycl-targets=%sycl_triple %s -o %t.out -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_60
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include "reduction_utils.hpp"
#include <iostream>

// This test performs basic checks of parallel_for(range<1>, reduction, func)
// with reductions initialized with a one element buffer and
// an initialize_to_identity property.

using namespace sycl;

constexpr bool Enable64Bit = ENABLE_64_BIT;

int NumErrors = 0;

template <typename Name, typename T, class BinaryOperation>
void tests(queue &Q, T Identity, T Init, BinaryOperation BOp, size_t NWItems) {
  NumErrors +=
      test<Name>(Q, Identity, Init, BOp, range<1>{NWItems}, init_to_identity());
}

int main() {
  queue Q;
  printDeviceInfo(Q);
  size_t MaxWGSize =
      Q.get_device().get_info<info::device::max_work_group_size>();

  // Fast-reduce and Fast-atomics. Try various range types/sizes.
  tests<class A1, int>(Q, 0, 99, std::plus<>{}, 1);
  tests<class A2, int>(Q, 0, 99, std::plus<>{}, 2);
  if constexpr (Enable64Bit) {
    tests<class A3, int64_t>(Q, 0, 99, std::plus<>{}, 7);
    tests<class A4, int64_t>(Q, 0, 99, std::plus<>{}, 64);
  }
  tests<class A5, int>(Q, 0, 99, std::plus<>{}, MaxWGSize * 2);
  tests<class A6, int>(Q, 0, 99, std::plus<>{}, MaxWGSize * 2 + 5);

  // Try various types & ranges.
  tests<class B1, int>(Q, ~0, 0xfefefefe, std::bit_and<>{}, 7);
  tests<class B2, int>(Q, 0, 0xfedcff99, std::bit_xor<>{}, MaxWGSize);
  tests<class B3, int>(Q, 0, 0xfedcff99, std::bit_or<>{}, 3);
  tests<class B4, short>(Q, 1, 2, std::multiplies<>{}, 7);
  tests<class B5, int>(Q, ~0, ~0, std::bit_and<>{}, 8);
  tests<class B6, int>(Q, 0, 0x12340000, std::bit_xor<>{}, 16);
  tests<class B7, int>(Q, 0, 0x3400, std::bit_or<>{}, MaxWGSize * 4);
  if constexpr (Enable64Bit)
    tests<class B8, uint64_t>(Q, 1, 2, std::multiplies<>{}, 31);
  tests<class B9, int>(Q, (std::numeric_limits<int>::max)(), -99,
                       ext::oneapi::minimum<>{}, MaxWGSize * 2);
  tests<class B10, int>(Q, (std::numeric_limits<int>::min)(), 99,
                        ext::oneapi::maximum<>{}, 8);
  tests<class B11, float>(Q, 1, 99, std::multiplies<>{}, 10);

  // Check with CUSTOM type.
  using CV = CustomVec<long long>;
  tests<class C1>(Q, CV(0), CV(99), CustomVecPlus<long long>{}, 64);
  tests<class C2>(Q, CV(0), CV(99), CustomVecPlus<long long>{}, 256);
  tests<class C3>(Q, CV(0), CV(99), CustomVecPlus<long long>{}, MaxWGSize * 3);

  printFinalStatus(NumErrors);
  return NumErrors;
}
