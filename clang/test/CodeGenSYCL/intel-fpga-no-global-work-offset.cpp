// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -sycl-std=2020 -emit-llvm -o - %s | FileCheck %s

#include "sycl.hpp"

using namespace sycl;
queue q;

class Foo {
public:
  [[intel::no_global_work_offset(1)]] void operator()() const {}
};

template <int SIZE>
class Functor {
public:
  [[intel::no_global_work_offset(SIZE)]] void operator()() const {}
};

int main() {
  q.submit([&](handler &h) {
    Foo boo;
    h.single_task<class kernel_name1>(boo);

    h.single_task<class kernel_name2>(
        []() [[intel::no_global_work_offset]]{});

    h.single_task<class kernel_name3>(
        []() [[intel::no_global_work_offset(0)]]{});

    Functor<1> f;
    h.single_task<class kernel_name4>(f);
  });
  return 0;
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name1() #0 {{.*}} !no_global_work_offset ![[NUM5:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name2() #0 {{.*}} !no_global_work_offset ![[NUM5]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name3() #0 {{.*}} ![[NUM4:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name4() #0 {{.*}} !no_global_work_offset ![[NUM5]]
// CHECK-NOT: ![[NUM4]]  = !{i32 0}
// CHECK: ![[NUM5]] = !{}
