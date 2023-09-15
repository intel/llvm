//  REQUIRES: cuda

// RUN: not %clangxx -ferror-limit=100 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_70 -fsyntax-only %s -o - 2>&1 | FileCheck %s

// NOTE: we can not use the `-verify` run alongside
// `expected-error`/`expected-warnings` as the diagnostics come from the device
// compilation, which happen in temporary files, while `expected-...` are
// placed in the main file, causing clang to complain at the file mismatch

#include <sycl/sycl.hpp>

template <int N1, int N2, int N3> class Functor {
public:
  [[intel::max_work_group_size(1, 1, N1), intel::min_work_groups_per_cu(N2),
    intel::max_work_groups_per_mp(N3)]] void
  // CHECK: maxclusterrank requires sm_90 or higher, CUDA arch provided: sm_70, ignoring 'max_work_groups_per_mp' attribute
  operator()() const {}
};

int main() {
  sycl::queue Q{};

  sycl::range<1> Gws(32);
  sycl::range<1> Lws(32);

  Q.submit([&](sycl::handler &cgh) {
     cgh.single_task<class T1>(
         sycl::nd_range<1>(Gws, Lws),
         [=]()
             [[intel::max_work_group_size(1, 1, 256),
               intel::min_work_groups_per_cu(2),
               // CHECK: maxclusterrank requires sm_90 or higher, CUDA arch provided: sm_70, ignoring 'max_work_groups_per_mp' attribute
               intel::max_work_groups_per_mp(4)]] { volatile int A = 42; });

     constexpr float A = 2.0;
     cgh.single_task<class T2>(
         [=]()
             [[intel::max_work_group_size(1, 1, 256),
               intel::min_work_groups_per_cu(A),
               // CHECK: 'min_work_groups_per_cu' attribute requires parameter 0 to be an integer constant
               intel::max_work_groups_per_mp(4)]] { volatile int A = 42; });
     // CHECK: maxclusterrank requires sm_90 or higher, CUDA arch provided: sm_70, ignoring 'max_work_groups_per_mp' attribute

     cgh.single_task<class T3>(
         [=]() [[intel::max_work_group_size(1, 1, 256),
                 intel::min_work_groups_per_cu(2147483647 + 1)]]
         // CHECK: 'min_work_groups_per_cu' attribute requires parameter 0 to be an integer constant
         { volatile int A = 42; });

     cgh.single_task<class T4>([=]() [[intel::max_work_group_size(1, 1, 256),
                                       intel::min_work_groups_per_cu(4),
                                       intel::min_work_groups_per_cu(8)]] {
       // CHECK: attribute 'min_work_groups_per_cu' is already applied with different arguments
       // CHECK: note: previous attribute is here
       volatile int A = 42;
     });

     cgh.single_task<class T5>([=]() [[intel::min_work_groups_per_cu(-8)]] {
       // CHECK: 'min_work_groups_per_cu' attribute must be greater than 0
       volatile int A = 42;
     });
   }).wait_and_throw();

  Q.single_task<class F>(Functor<512, 8, 16>{}).wait();

  return 0;
}
