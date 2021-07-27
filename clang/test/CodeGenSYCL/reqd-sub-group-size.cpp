// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -disable-llvm-passes -sycl-std=2017 -triple spir64-unknown-unknown-sycldevice -emit-llvm -o - %s | FileCheck %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

class Functor16 {
public:
  [[intel::reqd_sub_group_size(16)]] void operator()() const {}
};

[[intel::reqd_sub_group_size(8)]] void foo() {}

class Functor8 {
public:
  void operator()() const {
    foo();
  }
};

template <int SIZE>
class Functor2 {
public:
  [[intel::reqd_sub_group_size(SIZE)]] void operator()() const {}
};

template <int N>
[[intel::reqd_sub_group_size(N)]] void func() {}

int main() {
  q.submit([&](handler &h) {
    Functor16 f16;
    h.single_task<class kernel_name1>(f16);

    Functor8 f8;
    h.single_task<class kernel_name2>(f8);

    h.single_task<class kernel_name3>(
        []() [[intel::reqd_sub_group_size(4)]]{});

    Functor2<2> f2;
    h.single_task<class kernel_name4>(f2);

    h.single_task<class kernel_name5>([]() {
      func<2>();
    });
  });
  return 0;
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name1() #0 {{.*}} !intel_reqd_sub_group_size ![[SGSIZE16:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name2() #0 {{.*}} !intel_reqd_sub_group_size ![[SGSIZE8:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name3() #0 {{.*}} !intel_reqd_sub_group_size ![[SGSIZE4:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name4() #0 {{.*}} !intel_reqd_sub_group_size ![[SGSIZE2:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name5() #0 {{.*}} !intel_reqd_sub_group_size ![[SGSIZE2]]
// CHECK: ![[SGSIZE16]] = !{i32 16}
// CHECK: ![[SGSIZE8]] = !{i32 8}
// CHECK: ![[SGSIZE4]] = !{i32 4}
// CHECK: ![[SGSIZE2]] = !{i32 2}
