// RUN: %{build} -o %t.out %if target-nvidia %{ -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_60 %}
// RUN: %{run} %t.out

// This test performs basic checks of parallel_for(range<3>, reduction, func)
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

  tests<class A1, int>(Q, 0, 99, std::plus<>{}, range<3>{1, 1, 1});
  tests<class A2, int>(Q, 0, 99, std::plus<>{}, range<3>{2, 2, 2});
  tests<class A3, int>(Q, 0, 99, std::plus<>{}, range<3>{2, 3, 4});

  tests<class A4, int64_t>(Q, 0, 99, std::plus<>{},
                           range<3>{1, 1, MaxWGSize + 1});
  tests<class A5, int64_t>(Q, 0, 99, std::plus<>{},
                           range<3>{1, MaxWGSize + 1, 1});
  tests<class A6, int64_t>(Q, 0, 99, std::plus<>{},
                           range<3>{MaxWGSize + 1, 1, 1});

  tests<class A7, int64_t>(Q, 0, 99, std::plus<>{},
                           range<3>{2, 5, MaxWGSize * 2});
  tests<class A8, int64_t>(Q, 0, 99, std::plus<>{},
                           range<3>{3, MaxWGSize * 3, 2});
  tests<class A9, int64_t>(Q, 0, 99, std::plus<>{},
                           range<3>{MaxWGSize * 3, 8, 4});

  tests<class B1, int>(Q, 0, 0x2021ff99, std::bit_xor<>{},
                       range<3>{2, MaxWGSize * 3, 3});
  tests<class B2, int>(Q, ~0, 99, std::bit_and<>{},
                       range<3>{MaxWGSize * 3, 4, 3});
  tests<class B3, int>(Q, 0, 99, std::bit_or<>{},
                       range<3>{2, 2, MaxWGSize * 3});
  tests<class B4, uint64_t>(Q, 1, 3, std::multiplies<>{}, range<3>{2, 3, 5});
  tests<class B5, uint64_t>(Q, 1, 3, std::multiplies<>{}, range<3>{5, 2, 3});
  tests<class B6, int>(Q, (std::numeric_limits<int>::max)(), -99,
                       ext::oneapi::minimum<>{}, range<3>{MaxWGSize, 8, 3});
  tests<class B7, int>(Q, (std::numeric_limits<int>::min)(), 99,
                       ext::oneapi::maximum<>{}, range<3>{3, MaxWGSize, 3});
  tests<class B8, float>(Q, 1, 99, std::multiplies<>{}, range<3>{3, 3, 4});

  tests<class C1, CustomVec<long long>>(Q, 0, 99, CustomVecPlus<long long>{},
                                        range<3>{2, 33, MaxWGSize});
  tests<class C2, CustomVec<long long>>(Q, 99, CustomVecPlus<long long>{},
                                        range<3>{2, 33, MaxWGSize});

  tests<class D1, int>(Q, 99, PlusWithoutIdentity<int>{}, range<3>{1, 1, 1});
  tests<class D2, int>(Q, 99, PlusWithoutIdentity<int>{}, range<3>{2, 2, 2});
  tests<class D3, int>(Q, 99, PlusWithoutIdentity<int>{}, range<3>{2, 3, 4});

  tests<class D4, int64_t>(Q, 99, PlusWithoutIdentity<int64_t>{},
                           range<3>{1, 1, MaxWGSize + 1});
  tests<class D5, int64_t>(Q, 99, PlusWithoutIdentity<int64_t>{},
                           range<3>{1, MaxWGSize + 1, 1});
  tests<class D6, int64_t>(Q, 99, PlusWithoutIdentity<int64_t>{},
                           range<3>{MaxWGSize + 1, 1, 1});

  tests<class D7, int64_t>(Q, 99, PlusWithoutIdentity<int64_t>{},
                           range<3>{2, 5, MaxWGSize * 2});
  tests<class D8, int64_t>(Q, 99, PlusWithoutIdentity<int64_t>{},
                           range<3>{3, MaxWGSize * 3, 2});
  tests<class D9, int64_t>(Q, 99, PlusWithoutIdentity<int64_t>{},
                           range<3>{MaxWGSize * 3, 8, 4});

  tests<class E1, int>(Q, 99, MultipliesWithoutIdentity<int>{},
                       range<3>{1, 1, 1});
  tests<class E2, int>(Q, 99, MultipliesWithoutIdentity<int>{},
                       range<3>{2, 2, 2});
  tests<class E3, int>(Q, 99, MultipliesWithoutIdentity<int>{},
                       range<3>{2, 3, 4});

  tests<class E4, int64_t>(Q, 99, MultipliesWithoutIdentity<int64_t>{},
                           range<3>{1, 1, MaxWGSize + 1});
  tests<class E5, int64_t>(Q, 99, MultipliesWithoutIdentity<int64_t>{},
                           range<3>{1, MaxWGSize + 1, 1});
  tests<class E6, int64_t>(Q, 99, MultipliesWithoutIdentity<int64_t>{},
                           range<3>{MaxWGSize + 1, 1, 1});

  tests<class E7, int64_t>(Q, 99, MultipliesWithoutIdentity<int64_t>{},
                           range<3>{2, 5, MaxWGSize * 2});
  tests<class E8, int64_t>(Q, 99, MultipliesWithoutIdentity<int64_t>{},
                           range<3>{3, MaxWGSize * 3, 2});
  tests<class E9, int64_t>(Q, 99, MultipliesWithoutIdentity<int64_t>{},
                           range<3>{MaxWGSize * 3, 8, 4});

  printFinalStatus(NumErrors);
  return NumErrors;
}
