// RUN: %{build} -o %t.out %if any-device-is-cuda %{ -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_60 %}
// RUN: %{run} %t.out

// This test performs basic checks of parallel_for(range<2>, reduction, func)
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

  tests<class A1, int>(Q, 0, 99, std::plus<>{}, range<2>{1, 1});
  tests<class A2, int>(Q, 0, 99, std::plus<>{}, range<2>{2, 2});
  tests<class A3, int>(Q, 0, 99, std::plus<>{}, range<2>{2, 3});
  tests<class A4, int>(Q, 0, 99, std::plus<>{}, range<2>{MaxWGSize, 1});
  tests<class A5, int64_t>(Q, 0, 99, std::plus<>{}, range<2>{1, MaxWGSize});
  tests<class A6, int64_t>(Q, 0, 99, std::plus<>{}, range<2>{2, MaxWGSize * 2});
  tests<class A7, int64_t>(Q, 0, 99, std::plus<>{}, range<2>{MaxWGSize * 3, 7});
  tests<class A8, int64_t>(Q, 0, 99, std::plus<>{}, range<2>{3, MaxWGSize * 3});

  tests<class B1, CustomVec<long long>>(Q, 0, 99, CustomVecPlus<long long>{},
                                        range<2>{33, MaxWGSize});
  tests<class B2, CustomVec<long long>>(Q, 99, CustomVecPlus<long long>{},
                                        range<2>{33, MaxWGSize});

  tests<class C1, int>(Q, 99, PlusWithoutIdentity<int>{}, range<2>{1, 1});
  tests<class C2, int>(Q, 99, PlusWithoutIdentity<int>{}, range<2>{2, 2});
  tests<class C3, int>(Q, 99, PlusWithoutIdentity<int>{}, range<2>{2, 3});
  tests<class C4, int>(Q, 99, PlusWithoutIdentity<int>{},
                       range<2>{MaxWGSize, 1});
  tests<class C5, int64_t>(Q, 99, PlusWithoutIdentity<int64_t>{},
                           range<2>{1, MaxWGSize});
  tests<class C6, int64_t>(Q, 99, PlusWithoutIdentity<int64_t>{},
                           range<2>{2, MaxWGSize * 2});
  tests<class C7, int64_t>(Q, 99, PlusWithoutIdentity<int64_t>{},
                           range<2>{MaxWGSize * 3, 7});
  tests<class C8, int64_t>(Q, 99, PlusWithoutIdentity<int64_t>{},
                           range<2>{3, MaxWGSize * 3});

  printFinalStatus(NumErrors);
  return NumErrors;
}
