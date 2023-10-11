// REQUIRES: cuda

// RUN: %clangxx -ferror-limit=100 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_70 -fsycl-device-only -fsyntax-only -Xclang -verify %s

#include <sycl/sycl.hpp>

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
     // expected-warning@+5 {{'maxclusterrank' requires sm_90 or higher, CUDA arch provided: sm_70, ignoring 'max_work_groups_per_mp' attribute}}
     cgh.single_task<class T1>(
         [=]()
             [[intel::max_work_group_size(1, 1, 256),
               intel::min_work_groups_per_cu(2),
               intel::max_work_groups_per_mp(4)]] { volatile int A = 42; });

     constexpr float A = 2.0;
     // expected-error@+5 {{'min_work_groups_per_cu' attribute requires parameter 0 to be an integer constant}}
     // expected-warning@+5 {{'maxclusterrank' requires sm_90 or higher, CUDA arch provided: sm_70, ignoring 'max_work_groups_per_mp' attribute}}
     cgh.single_task<class T2>(
         [=]()
             [[intel::max_work_group_size(1, 1, 256),
               intel::min_work_groups_per_cu(A),
               intel::max_work_groups_per_mp(4)]] { volatile int A = 42; });

     // expected-error@+3 {{'min_work_groups_per_cu' attribute requires parameter 0 to be an integer constant}}
     cgh.single_task<class T3>(
         [=]() [[intel::max_work_group_size(1, 1, 256),
                 intel::min_work_groups_per_cu(2147483647 + 1)]]
         { volatile int A = 42; });

     // expected-warning@+4 {{attribute 'min_work_groups_per_cu' is already applied with different arguments}}
     // expected-note@+2 {{previous attribute is here}}
     cgh.single_task<class T4>([=]() [[intel::max_work_group_size(1, 1, 256),
                                       intel::min_work_groups_per_cu(4),
                                       intel::min_work_groups_per_cu(8)]] {
       volatile int A = 42;
     });

     // expected-error@+1 {{'min_work_groups_per_cu' attribute must be greater than 0}}
     cgh.single_task<class T5>([=]() [[intel::min_work_groups_per_cu(-8)]] {
       volatile int A = 42;
     });
   }).wait_and_throw();

  Q.single_task<class F>(Functor<512, 8, 16>{}).wait();

  return 0;
}
