// RUN: %clang_cc1 -internal-isystem %S/Inputs %s -triple nvptx64-nvidia-cuda -target-cpu sm_90 -fsycl-is-device -fsyntax-only -Wno-c++23-extensions -verify
// expected-no-diagnostics

// Maximum work groups per multi-processor, mapped to maxclusterrank PTX
// directive, is an SM_90 feature, make sure that no warnings/errors are issued.

#include "sycl.hpp"

template <int N1, int N2, int N3> class Functor {
public:
  [[intel::max_work_group_size(1, 1, N1), intel::min_work_groups_per_cu(N2),
    intel::max_work_groups_per_mp(N3)]] void
  operator()() const {}
};

int main() {
  sycl::queue Q{};

  Q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class T1>( [=] [[intel::max_work_group_size(1, 1, 256),
                                          intel::min_work_groups_per_cu(2),
                                          intel::max_work_groups_per_mp(4)]] (
) { volatile int A = 42; });
  });

  Q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class F>(Functor<512, 8, 16>{});
  });

  return 0;
}
