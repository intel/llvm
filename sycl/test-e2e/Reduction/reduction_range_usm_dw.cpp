// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_60
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// Error message `Group algorithms are not
// supported on host device.` on Nvidia.
// XFAIL: hip_nvidia

#include "reduction_utils.hpp"

// This test performs basic checks of parallel_for(range<1>, reduction, func)
// with reductions initialized a USM pointer and an initialize_to_identity
// property.

using namespace sycl;

int NumErrors = 0;

template <typename Name, typename T, class BinaryOperation, int Dims>
void tests(queue &Q, T Identity, T Init, BinaryOperation BOp,
           const range<Dims> &Range) {
  auto PropList = init_to_identity();
  NumErrors += testUSM<TName<Name, class Shared>, T>(
      Q, Identity, Init, BOp, Range, usm::alloc::shared, PropList);
  NumErrors += testUSM<TName<Name, class Host>, T>(
      Q, Identity, Init, BOp, Range, usm::alloc::host, PropList);
  NumErrors += testUSM<TName<Name, class Device>, T>(
      Q, Identity, Init, BOp, Range, usm::alloc::device, PropList);
}

int main() {
  queue Q;
  printDeviceInfo(Q);
  size_t MaxWGSize =
      Q.get_device().get_info<info::device::max_work_group_size>();

  // Fast-reduce and Fast-atomics. Try various range types/sizes.
  tests<class A1, int>(Q, 0, 99, std::plus<>{}, range<1>{MaxWGSize * 2 + 5});
  tests<class A2, float>(Q, 0, 99, std::plus<>{}, range<2>{1, 1});
  tests<class A3, int>(Q, 0, 99, std::plus<>{}, range<3>{MaxWGSize, 1, 2});

  // Try various operations.
  tests<class B1, int>(Q, ~0, 99, std::bit_and<>{}, range<1>{8});
  tests<class B2, int>(Q, 0, 0xff99, std::bit_xor<>{}, range<1>{MaxWGSize + 1});
  tests<class B4, short>(Q, 1, 2, std::multiplies<>{}, range<1>{7});
  tests<class B5, int>(Q, (std::numeric_limits<int>::max)(), -99,
                       ext::oneapi::minimum<>{}, range<2>{MaxWGSize, 2});

  // Check with CUSTOM type.
  tests<class C1>(Q, CustomVec<long long>(0), CustomVec<long long>(99),
                  CustomVecPlus<long long>{}, range<2>{3, MaxWGSize});

  printFinalStatus(NumErrors);
  return NumErrors;
}
