// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -disable-llvm-passes -triple spir64-unknown-unknown-sycldevice -Wno-sycl-2017-compat -emit-llvm -o - %s | FileCheck %s

#include "sycl.hpp"

// CHECK: br label %for.cond,   !llvm.loop ![[MD_MC:[0-9]+]]
// CHECK: br label %for.cond2,  !llvm.loop ![[MD_MC_1:[0-9]+]]

// CHECK: define {{.*}}spir_kernel void @"{{.*}}kernel_name1"() #0 {{.*}} !max_concurrency !16
// CHECK: define {{.*}}spir_kernel void @"{{.*}}kernel_name2"() #0 {{.*}} !max_concurrency ![[NUM2:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @"{{.*}}kernel_name3"() #0 {{.*}} !max_concurrency ![[NUM3:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @"{{.*}}kernel_name4"() #0 {{.*}} !max_concurrency ![[NUM1:[0-9]+]]

template <int A>
void max_concurrency() {
  int a[10];
  // CHECK: ![[MD_MC]] = distinct !{![[MD_MC]], ![[MP:[0-9]+]], ![[MD_max_concurrency:[0-9]+]]}
  // CHECK-NEXT: ![[MP]] = !{!"llvm.loop.mustprogress"}
  // CHECK-NEXT: ![[MD_max_concurrency]] = !{!"llvm.loop.max_concurrency.count", i32 5}
  [[intel::max_concurrency(A)]] for (int i = 0; i != 10; ++i)
    a[i] = 0;
  // CHECK: ![[MD_MC_1]] = distinct !{![[MD_MC_1]], ![[MP]], ![[MD_max_concurrency_1:[0-9]+]]}
  // CHECK-NEXT: ![[MD_max_concurrency_1]] = !{!"llvm.loop.max_concurrency.count", i32 4}
  [[intel::max_concurrency(4)]] for (int i = 0; i != 10; ++i)
      a[i] = 0;
}

// CHECK: !16 = !{i32 4}
// CHECK: !17 = !{i32 2}
// CHECK: !18 = !{i32 3}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task_1(const Func &kernelFunc) {
  kernelFunc();
}

using namespace cl::sycl;
queue q;

class Functor1 {
public:
  [[intel::max_concurrency(4)]] void operator()() const {}
};

[[intel::max_concurrency(2)]] void foo() {}

class Functor2 {
public:
  void operator()() const {
    foo();
  }
};

template <int NT>
class Functor3 {
public:
  [[intel::max_concurrency(NT)]] void operator()() const {}
};

template <int NT>
[[intel::reqd_sub_group_size(NT)]] void func() {}

int main() {
  kernel_single_task_1<class kernel_function>([]() {
     max_concurrency<5>();
   });

  q.submit([&](handler &h) {
    Functor1 f1;
    h.single_task<class kernel_name1>(f1);

    Functor2 f2;
    h.single_task<class kernel_name2>(f2);


    h.single_task<class kernel_name3>(
        []() [[intel::max_concurrency(3)]]{});

    Functor3<4> f3;
    h.single_task<class kernel_name4>(f3);

    h.single_task<class kernel_name5>([]() {
      func<2>();
    });

  });


  return 0;
}


