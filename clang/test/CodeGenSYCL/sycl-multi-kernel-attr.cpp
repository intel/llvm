// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -sycl-std=2017 -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

class Functor {
public:
  [[intel::reqd_sub_group_size(4), cl::reqd_work_group_size(32, 16, 16)]] void operator()() const {}
};

class Functor1 {
public:
  [[intel::reqd_sub_group_size(2), sycl::reqd_work_group_size(64, 32, 32)]] void operator()() const {}
};

template <int SIZE, int SIZE1, int SIZE2>
class Functor2 {
public:
  [[sycl::reqd_work_group_size(SIZE, SIZE1, SIZE2)]] void operator()() const {}
};

template <int N, int N1, int N2>
[[sycl::reqd_work_group_size(N, N1, N2)]] void func() {}

int main() {
  q.submit([&](handler &h) {
    Functor foo;
    h.single_task<class kernel_name1>(foo);

    Functor1 foo1;
    h.single_task<class kernel_name2>(foo1);

    Functor2<2, 2, 2> foo2;
    h.single_task<class kernel_name3>(foo2);

    h.single_task<class kernel_name4>([]() {
      func<8, 4, 4>();
    });
  });
  return 0;
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name1() #0 {{.*}} !reqd_work_group_size ![[WGSIZE:[0-9]+]] !intel_reqd_sub_group_size ![[SGSIZE:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name2() #0 {{.*}} !reqd_work_group_size ![[WGSIZE1:[0-9]+]] !intel_reqd_sub_group_size ![[SGSIZE1:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name3() #0 {{.*}} !reqd_work_group_size ![[WGSIZE2:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name4() #0 {{.*}} !reqd_work_group_size ![[WGSIZE3:[0-9]+]]
// CHECK: ![[WGSIZE]] = !{i32 16, i32 16, i32 32}
// CHECK: ![[SGSIZE]] = !{i32 4}
// CHECK: ![[WGSIZE1]] = !{i32 32, i32 32, i32 64}
// CHECK: ![[SGSIZE1]] = !{i32 2}
// CHECK: ![[WGSIZE2]] = !{i32 2, i32 2, i32 2}
// CHECK: ![[WGSIZE3]] = !{i32 4, i32 4, i32 8}
