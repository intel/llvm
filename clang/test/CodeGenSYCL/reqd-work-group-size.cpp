// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -sycl-std=2017 -emit-llvm -o - %s | FileCheck %s

#include "sycl.hpp"

using namespace cl::sycl;
queue q;

class Functor32x16x16 {
public:
  [[cl::reqd_work_group_size(32, 16, 16)]] void operator()() const {}
};

[[cl::reqd_work_group_size(8, 1, 1)]] void f8x1x1() {}

class Functor {
public:
  void operator()() const {
    f8x1x1();
  }
};

template <int SIZE, int SIZE1, int SIZE2>
class FunctorTemp {
public:
  [[cl::reqd_work_group_size(SIZE, SIZE1, SIZE2)]] void operator()() const {}
};

template <int N, int N1, int N2>
[[cl::reqd_work_group_size(N, N1, N2)]] void func() {}

int main() {
  q.submit([&](handler &h) {
    Functor32x16x16 f32x16x16;
    h.single_task<class kernel_name1>(f32x16x16);

    Functor f;
    h.single_task<class kernel_name2>(f);

    h.single_task<class kernel_name3>(
        []() [[cl::reqd_work_group_size(8, 8, 8)]]{});

    FunctorTemp<2, 2, 2> ft;
    h.single_task<class kernel_name4>(ft);

    h.single_task<class kernel_name5>([]() {
      func<8, 4, 4>();
    });

    h.single_task<class kernel_name6>(
        []() [[cl::reqd_work_group_size(1, 8, 2)]]{});
  });
  return 0;
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name1() #0 {{.*}} !reqd_work_group_size ![[WGSIZE32:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name2() #0 {{.*}} !reqd_work_group_size ![[WGSIZE8:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name3() #0 {{.*}} !reqd_work_group_size ![[WGSIZE88:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name4() #0 {{.*}} !reqd_work_group_size ![[WGSIZE22:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name5() #0 {{.*}} !reqd_work_group_size ![[WGSIZE44:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name6() #0 {{.*}} !reqd_work_group_size ![[WGSIZE2:[0-9]+]]
// CHECK: ![[WGSIZE32]] = !{i32 16, i32 16, i32 32}
// CHECK: ![[WGSIZE8]] = !{i32 1, i32 1, i32 8}
// CHECK: ![[WGSIZE88]] = !{i32 8, i32 8, i32 8}
// CHECK: ![[WGSIZE22]] = !{i32 2, i32 2, i32 2}
// CHECK: ![[WGSIZE44]] = !{i32 4, i32 4, i32 8}
// CHECK: ![[WGSIZE2]] = !{i32 2, i32 8, i32 1}
