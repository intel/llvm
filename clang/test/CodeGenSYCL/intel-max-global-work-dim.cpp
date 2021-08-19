// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -sycl-std=2017 -emit-llvm -o - %s | FileCheck %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

class Foo {
public:
  [[intel::max_global_work_dim(1)]] void operator()() const {}
};

template <int SIZE>
class Functor {
public:
  [[intel::max_global_work_dim(SIZE)]] void operator()() const {}
};

template <int N>
[[intel::max_global_work_dim(N)]] void func() {}

int main() {
  q.submit([&](handler &h) {
    Foo boo;
    h.single_task<class kernel_name1>(boo);

    h.single_task<class kernel_name2>(
        []() [[intel::max_global_work_dim(2)]]{});

    Functor<2> f;
    h.single_task<class kernel_name3>(f);

    h.single_task<class kernel_name4>([]() {
      func<2>();
    });
  });
  return 0;
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name1() #0 {{.*}} !max_global_work_dim ![[NUM1:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name2() #0 {{.*}} !max_global_work_dim ![[NUM2:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name3() #0 {{.*}} !max_global_work_dim ![[NUM2]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name4() #0 {{.*}} !max_global_work_dim ![[NUM2]]
// CHECK: ![[NUM1]] = !{i32 1}
// CHECK: ![[NUM2]] = !{i32 2}
