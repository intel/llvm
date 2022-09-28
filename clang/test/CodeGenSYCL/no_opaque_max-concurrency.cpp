// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -disable-llvm-passes -triple spir64-unknown-unknown -sycl-std=2020 -no-opaque-pointers -emit-llvm -o - %s | FileCheck %s

#include "sycl.hpp"

// CHECK-LABEL: define{{.*}}@_Z15max_concurrencyILi5EEvv()
// CHECK: entry:
// CHECK: [[A:%.*]] = alloca [10 x i32], align 4
// CHECK: [[A_CAST:%.*]] = addrspacecast [10 x i32]* [[A]] to [10 x i32] addrspace(4)*
// CHECK: %4 = load i32, i32 addrspace(4)* %i.ascast, align 4
// CHECK: [[IDXPROM:%*]] = sext i32 %4 to i64
// CHECK: %arrayidx = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* [[A_CAST]], i64 0, i64 [[IDXPROM]]
// CHECK: store i32 0, i32 addrspace(4)* %arrayidx, align 4
// CHECK: [[TMP2:%.*]] = load i32, i32 addrspace(4)* %i.ascast, align 4
// CHECK: %inc = add nsw i32 [[TMP2]], 1
// CHECK: store i32 %inc, i32 addrspace(4)* %i.ascast, align 4
// CHECK: br label %for.cond,   !llvm.loop ![[MD_MC:[0-9]+]]
// CHECK: store i32 %inc10, i32 addrspace(4)* %i1.ascast, align 4
// CHECK: br label %for.cond2,  !llvm.loop ![[MD_MC_1:[0-9]+]]
// CHECK: ret void

// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name1() [[ATTR0:#.*]] {{.*}} !max_concurrency ![[NUM1:[0-9]+]]

// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name4() [[ATTR0]] {{.*}} !max_concurrency ![[NUM1:[0-9]+]]

// CHECK: define dso_local spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_E12kernel_name5()

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

// CHECK: ![[NUM1]] = !{i32 4}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task_1(const Func &kernelFunc) {
  kernelFunc();
}

using namespace sycl;

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
[[intel::max_concurrency(NT)]] void func() {}

int main() {
  queue q;

  kernel_single_task_1<class kernel_function>([]() {
     max_concurrency<5>();
   });

  q.submit([&](handler &h) {
    Functor1 f1;
    h.single_task<class kernel_name1>(f1);

    Functor2 f2;
    h.single_task<class kernel_name2>(f2);

    Functor3<4> f3;
    h.single_task<class kernel_name4>(f3);

    h.single_task<class kernel_name5>([]() {
      func<2>();
    });

  });


  return 0;
}
