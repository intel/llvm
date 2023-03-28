// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_60
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// This test performs basic checks of parallel_for(range<1>, reduction, func)
// with reductions initialized with a one element buffer.

#include "reduction_utils.hpp"

using namespace sycl;

int NumErrors = 0;

template <typename Name, typename T, typename... ArgTys>
void tests(ArgTys &&...Args) {
  NumErrors += test<Name, T>(std::forward<ArgTys>(Args)...);
}

int main() {
  queue Q;
  printDeviceInfo(Q);
  size_t MaxWGSize =
      Q.get_device().get_info<info::device::max_work_group_size>();

  constexpr access::mode RW = access::mode::read_write;
  // Fast-reduce and Fast-atomics. Try various range types/sizes.
  tests<class A1, int>(Q, 0, 99, std::plus<int>{}, range<1>(1));
  tests<class A2, int>(Q, 0, 99, std::plus<>{}, range<1>(2));
  tests<class A3, int>(Q, 0, 99, std::plus<>{}, range<1>(7));
  tests<class A4, int>(Q, 0, 99, std::plus<>{}, range<1>(64));
  tests<class A5, int>(Q, 0, 99, std::plus<>{}, range<1>(MaxWGSize * 2));
  tests<class A6, int>(Q, 0, 99, std::plus<>{}, range<1>(MaxWGSize * 2 + 5));

  // Try various types & ranges.
  tests<class B1, int>(Q, ~0, ~0, std::bit_and<>{}, range<1>(8));
  tests<class B2, int>(Q, 0, 0x12340000, std::bit_xor<>{}, range<1>(16));
  tests<class B3, int>(Q, 0, 0x3400, std::bit_or<>{}, range<1>(MaxWGSize * 3));
  tests<class B4, uint64_t>(Q, 1, 2, std::multiplies<>{}, range<1>(16));
  tests<class B5, float>(Q, 1, 3, std::multiplies<>{}, range<1>(11));
  tests<class B6, int>(Q, (std::numeric_limits<int>::max)(), -99,
                       ext::oneapi::minimum<>{}, range<1>(MaxWGSize * 2));
  tests<class B7, int>(Q, (std::numeric_limits<int>::min)(), 99,
                       ext::oneapi::maximum<>{}, range<1>(8));

  // Check with CUSTOM type.
  tests<class C1, CustomVec<long long>>(Q, 0, 99, CustomVecPlus<long long>{},
                                        range<1>(256));
  tests<class C2, CustomVec<long long>>(Q, 0, 99, CustomVecPlus<long long>{},
                                        range<1>(MaxWGSize * 3));
  tests<class C3, CustomVec<long long>>(Q, 99, CustomVecPlus<long long>{},
                                        range<1>(72));

  // Check with identityless operations.
  tests<class D1, int>(Q, 99, PlusWithoutIdentity<int>{}, range<1>(1));
  tests<class D2, int>(Q, 99, PlusWithoutIdentity<int>{}, range<1>(2));
  tests<class D3, int>(Q, 99, PlusWithoutIdentity<int>{}, range<1>(7));
  tests<class D4, int>(Q, 99, PlusWithoutIdentity<int>{}, range<1>(64));
  tests<class D5, int>(Q, 99, PlusWithoutIdentity<int>{},
                       range<1>(MaxWGSize * 2));
  tests<class D6, int>(Q, 99, PlusWithoutIdentity<int>{},
                       range<1>(MaxWGSize * 2 + 5));

  printFinalStatus(NumErrors);
  return NumErrors;
}
