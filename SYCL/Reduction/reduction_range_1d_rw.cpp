// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_60
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test performs basic checks of parallel_for(range<1>, reduction, func)
// with reductions initialized with a one element buffer.

#include "reduction_utils.hpp"

using namespace sycl;

int NumErrors = 0;

int main() {
  queue Q;
  printDeviceInfo(Q);
  size_t MaxWGSize =
      Q.get_device().get_info<info::device::max_work_group_size>();

  constexpr access::mode RW = access::mode::read_write;
  // Fast-reduce and Fast-atomics. Try various range types/sizes.
  test<class A1, int>(Q, 0, 99, std::plus<int>{}, range<1>(1));
  test<class A2, int>(Q, 0, 99, std::plus<>{}, range<1>(2));
  test<class A3, int>(Q, 0, 99, std::plus<>{}, range<1>(7));
  test<class A4, int>(Q, 0, 99, std::plus<>{}, range<1>(64));
  test<class A5, int>(Q, 0, 99, std::plus<>{}, range<1>(MaxWGSize * 2));
  test<class A6, int>(Q, 0, 99, std::plus<>{}, range<1>(MaxWGSize * 2 + 5));

  // Try various types & ranges.
  test<class B1, int>(Q, ~0, ~0, std::bit_and<>{}, range<1>(8));
  test<class B2, int>(Q, 0, 0x12340000, std::bit_xor<>{}, range<1>(16));
  test<class B3, int>(Q, 0, 0x3400, std::bit_or<>{}, range<1>(MaxWGSize * 3));
  test<class B4, uint64_t>(Q, 1, 2, std::multiplies<>{}, range<1>(16));
  test<class B5, float>(Q, 1, 3, std::multiplies<>{}, range<1>(11));
  test<class B6, int>(Q, (std::numeric_limits<int>::max)(), -99,
                      ext::oneapi::minimum<>{}, range<1>(MaxWGSize * 2));
  test<class B7, int>(Q, (std::numeric_limits<int>::min)(), 99,
                      ext::oneapi::maximum<>{}, range<1>(8));

  // Check with CUSTOM type.
  using CV = CustomVec<long long>;
  test<class C1>(Q, CV(0), CV(99), CustomVecPlus<long long>{}, range<1>(256));
  test<class C2>(Q, CV(0), CV(99), CustomVecPlus<long long>{},
                 range<1>(MaxWGSize * 3));

  printFinalStatus(NumErrors);
  return NumErrors;
}
