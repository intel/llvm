// RUN: %{build} -o %t.out %if any-device-is-cuda %{ -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_60 %}
// RUN: %{run} %t.out

// This test performs basic checks of parallel_for(range<1>, reduction, func)
// with reductions initialized with a one element buffer. Additionally, some
// reducers will not be written to.

#include "reduction_utils.hpp"

using namespace sycl;

int NumErrors = 0;

template <typename T> class SkipEvenName;
template <typename T> class SkipOddName;
template <typename T> class SkipAllName;

template <typename Name, typename T, typename... ArgTys>
void tests(ArgTys &&...Args) {
  NumErrors += test<SkipEvenName<Name>, T>(std::forward<ArgTys>(Args)...,
                                           property_list{}, SkipEvenOp{});
  NumErrors += test<SkipOddName<Name>, T>(std::forward<ArgTys>(Args)...,
                                          property_list{}, SkipOddOp{});
  NumErrors += test<SkipAllName<Name>, T>(std::forward<ArgTys>(Args)...,
                                          property_list{}, SkipAllOp{});
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

  // Check with CUSTOM type.
  tests<class B1, CustomVec<long long>>(Q, 0, 99, CustomVecPlus<long long>{},
                                        range<1>(256));
  tests<class B2, CustomVec<long long>>(Q, 0, 99, CustomVecPlus<long long>{},
                                        range<1>(MaxWGSize * 3));
  tests<class B3, CustomVec<long long>>(Q, 99, CustomVecPlus<long long>{},
                                        range<1>(72));

  // Check with identityless operations.
  tests<class C1, int>(Q, 99, PlusWithoutIdentity<int>{}, range<1>(1));
  tests<class C2, int>(Q, 99, PlusWithoutIdentity<int>{}, range<1>(2));
  tests<class C3, int>(Q, 99, PlusWithoutIdentity<int>{}, range<1>(7));
  tests<class C4, int>(Q, 99, PlusWithoutIdentity<int>{}, range<1>(64));
  tests<class C5, int>(Q, 99, PlusWithoutIdentity<int>{},
                       range<1>(MaxWGSize * 2));
  tests<class C6, int>(Q, 99, PlusWithoutIdentity<int>{},
                       range<1>(MaxWGSize * 2 + 5));

  printFinalStatus(NumErrors);
  return NumErrors;
}
