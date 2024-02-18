// RUN: %clang_cc1 -internal-isystem %S/Inputs -triple nvptx-unknown-unknown -target-cpu sm_70 -fsycl-is-device -Wno-c++23-extensions %s -o -fsyntax-only -verify %s

// Maximum work groups per multi-processor, mapped to maxclusterrank PTX
// directive, is an SM_90 feature, make sure that correct warning is issued on
// architectures lower than that. Furthermore, warn/error incorrect values
// specified for max_work_groups_per_mp and min_work_groups_per_cu.

#include "sycl.hpp"

template <int N1, int N2, int N3> class Functor {
public:
  // expected-warning@+2 {{'maxclusterrank' requires sm_90 or higher, CUDA arch provided: sm_70, ignoring 'max_work_groups_per_mp' attribute}}
  [[intel::max_work_group_size(1, 1, N1), intel::min_work_groups_per_cu(N2),
    intel::max_work_groups_per_mp(N3)]] void
  operator()() const {}
};

int main() {
  sycl::queue Q{};

  Q.submit([&](sycl::handler &cgh) {
    // expected-warning@+4 {{'maxclusterrank' requires sm_90 or higher, CUDA arch provided: sm_70, ignoring 'max_work_groups_per_mp' attribute}}
    cgh.single_task<class T1>(
        [=] [[intel::max_work_group_size(1, 1, 256),
              intel::min_work_groups_per_cu(2),
              intel::max_work_groups_per_mp(4)]] () { volatile int A = 42; });

    constexpr float A = 2.0;
    // expected-warning@+5{{'maxclusterrank' requires sm_90 or higher, CUDA arch provided: sm_70, ignoring 'max_work_groups_per_mp' attribute}}
    // expected-error@+3 {{integral constant expression must have integral or unscoped enumeration type, not 'float'}}
    cgh.single_task<class T2>(
        [=] [[intel::max_work_group_size(1, 1, 256),
              intel::min_work_groups_per_cu(A),
              intel::max_work_groups_per_mp(4)]] () { volatile int A = 42; });

    // expected-error@+4 {{expression is not an integral constant expression}}
    // expected-note@+3 {{value 2147483648 is outside the range of representable values of type 'int'}}
    cgh.single_task<class T3>(
        [=] [[intel::max_work_group_size(1, 1, 256),
              intel::min_work_groups_per_cu(2147483647 + 1)]] () {
          volatile int A = 42;
        });

    // expected-warning@+5 {{attribute 'min_work_groups_per_cu' is already applied with different arguments}}
    // expected-note@+3 {{previous attribute is here}}
    cgh.single_task<class T4>(
        [=] [[intel::max_work_group_size(1, 1, 256),
              intel::min_work_groups_per_cu(4),
              intel::min_work_groups_per_cu(8)]] () { volatile int A = 42; });

    // expected-error@+2 {{'min_work_groups_per_cu' attribute requires a non-negative integral compile time constant expression}}
    cgh.single_task<class T5>(
        [=] [[intel::min_work_groups_per_cu(-8)]] () { volatile int A = 42; });
  });

  Q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class F>(Functor<512, 8, 16>{});
  });

  return 0;
}
