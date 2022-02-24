// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_60
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// TODO: test disabled due to sporadic fails in level_zero:gpu RT.
// UNSUPPORTED: linux && level_zero

// This test performs basic checks of parallel_for(range<1>, reduction, func)
// with reductions initialized with 1-dimensional read_write accessor
// accessing 1 element buffer.

#include "reduction_range_scalar.hpp"

using namespace cl::sycl;

int NumErrors = 0;

template <typename Name, typename T, class BinaryOperation>
void tests(queue &Q, T Identity, T Init, BinaryOperation BOp, size_t NWItems) {
  constexpr access::mode RW = access::mode::read_write;
  NumErrors += testBoth<Name, RW>(Q, Identity, Init, BOp, range<1>{NWItems});
}

int main() {
  queue Q;
  printDeviceInfo(Q);
  size_t MaxWGSize =
      Q.get_device().get_info<info::device::max_work_group_size>();

  // Fast-reduce and Fast-atomics. Try various range types/sizes.
  tests<class A1, int>(Q, 0, 99, std::plus<int>{}, 1);
  tests<class A2, int>(Q, 0, 99, std::plus<>{}, 7);
  tests<class A3, int>(Q, 0, 99, std::plus<>{}, 64);
  tests<class A4, int>(Q, 0, 99, std::plus<>{}, MaxWGSize * 2);
  tests<class A5, int>(Q, 0, 99, std::plus<>{}, MaxWGSize * 2 + 5);

  // Try various types & ranges.
  tests<class B1, int>(Q, ~0, ~0, std::bit_and<>{}, 8);
  tests<class B2, int>(Q, 0, 0x12340000, std::bit_xor<>{}, 16);
  tests<class B3, int>(Q, 0, 0x3400, std::bit_or<>{}, MaxWGSize * 3);
  tests<class B4, uint64_t>(Q, 1, 2, std::multiplies<>{}, 16);
  tests<class B5, float>(Q, 1, 3, std::multiplies<>{}, 11);
  tests<class B6, int>(Q, (std::numeric_limits<int>::max)(), -99,
                       ext::oneapi::minimum<>{}, MaxWGSize * 2);
  tests<class B7, int>(Q, (std::numeric_limits<int>::min)(), 99,
                       ext::oneapi::maximum<>{}, 8);

  // Check with CUSTOM type.
  tests<class C1>(Q, CustomVec<long long>(0), CustomVec<long long>(99),
                  CustomVecPlus<long long>{}, 256);
  tests<class C2>(Q, CustomVec<long long>(0), CustomVec<long long>(99),
                  CustomVecPlus<long long>{}, MaxWGSize * 3);

  printFinalStatus(NumErrors);
  return NumErrors;
}
