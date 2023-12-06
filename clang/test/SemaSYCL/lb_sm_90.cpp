// RUN: %clang_cc1 -internal-isystem %S/Inputs %s -triple nvptx64-nvidia-cuda -target-cpu sm_90 -fsycl-is-device -fsyntax-only -Wno-c++23-extensions -verify -S -o %t

// Maximum work groups per multi-processor, mapped to maxclusterrank PTX
// directive, is an SM_90 feature. Attributes need to be used in sequence:
// max_work_group_size, min_work_groups_per_cu, max_work_groups_per_mp, warn on
// missing attributes in sequences.

#include "sycl.hpp"

template <int N1, int N2, int N3> class Functor {
public:
  [[intel::max_work_group_size(1, 1, N1), intel::min_work_groups_per_cu(N2),
    intel::max_work_groups_per_mp(N3)]] void
  operator()() const {}
};

// expected-warning@+1 {{'max_work_groups_per_mp' attribute ignored, as it requires: maximum work group size and minimum work groups per compute unit to be also specified}}
template <int N1, int N2> class Functor_2 {
public:
  [[intel::max_work_group_size(1, 1, N1),
    intel::max_work_groups_per_mp(N2)]] void
  operator()() const {}
};

int main() {
  sycl::queue Q{};

  Q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class T1>(
        [=] [[intel::max_work_group_size(1, 1, 256),
              intel::min_work_groups_per_cu(2),
              intel::max_work_groups_per_mp(4)]] () { volatile int A = 42; });

    // expected-warning@+2 {{'max_work_groups_per_mp' attribute ignored, as it requires: maximum work group size and minimum work groups per compute unit to be also specified}}
    cgh.single_task<class T2>(
        [=] [[intel::max_work_group_size(1, 1, 256),
              intel::max_work_groups_per_mp(4)]] () { volatile int A = 42; });

    // expected-warning@+2 {{'max_work_groups_per_mp' attribute ignored, as it requires: maximum work group size and minimum work groups per compute unit to be also specified}}
    cgh.single_task<class T3>(
        [=] [[intel::max_work_groups_per_mp(4)]] () { volatile int A = 42; });

    // expected-warning@+2 {{'min_work_groups_per_cu' attribute ignored, as it requires: maximum work group size to be also specified}} cgh.single_task<class T4>(
    cgh.single_task<class T4>(
        [=] [[intel::min_work_groups_per_cu(4)]] () { volatile int A = 42; });
  });

  Q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class F>(Functor<512, 8, 16>{});
  });

  Q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class F2>(Functor_2<512, 8>{});
  });

  return 0;
}
