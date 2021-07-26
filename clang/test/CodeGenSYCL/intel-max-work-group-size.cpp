// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -sycl-std=2017 -emit-llvm -o - %s | FileCheck %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

class Foo {
public:
  [[intel::max_work_group_size(1, 1, 1)]] void operator()() const {}
};

class Bar {
public:
  [[intel::max_work_group_size(1, 3, 6)]] void operator()() const {}
};

template <int SIZE, int SIZE1, int SIZE2>
class Functor {
public:
  [[intel::max_work_group_size(SIZE, SIZE1, SIZE2)]] void operator()() const {}
};

template <int N, int N1, int N2>
[[intel::max_work_group_size(N, N1, N2)]] void func() {}

int main() {
  q.submit([&](handler &h) {
    Foo boo;
    h.single_task<class kernel_name1>(boo);

    h.single_task<class kernel_name2>(
        []() [[intel::max_work_group_size(8, 8, 8)]]{});

    Bar bar;
    h.single_task<class kernel_name3>(bar);

    Functor<2, 2, 2> f;
    h.single_task<class kernel_name4>(f);

    h.single_task<class kernel_name5>([]() {
      func<4, 4, 4>();
    });
  });
  return 0;
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name1() #0 {{.*}} !max_work_group_size ![[NUM1:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name2() #0 {{.*}} !max_work_group_size ![[NUM8:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name3() #0 {{.*}} !max_work_group_size ![[NUM6:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name4() #0 {{.*}} !max_work_group_size ![[NUM2:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name5() #0 {{.*}} !max_work_group_size ![[NUM4:[0-9]+]]
// CHECK: ![[NUM1]] = !{i32 1, i32 1, i32 1}
// CHECK: ![[NUM8]] = !{i32 8, i32 8, i32 8}
// CHECK: ![[NUM6]] = !{i32 6, i32 3, i32 1}
// CHECK: ![[NUM2]] = !{i32 2, i32 2, i32 2}
// CHECK: ![[NUM4]] = !{i32 4, i32 4, i32 4}
