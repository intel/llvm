// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -sycl-std=2020 -emit-llvm -o - %s | FileCheck %s

#include "sycl.hpp"

using namespace sycl;
queue q;

class Functor32 {
public:
  [[sycl::reqd_work_group_size(32)]] void operator()() const {}
};
class Functor32x16 {
public:
  [[sycl::reqd_work_group_size(32, 16)]] void operator()() const {}
};
class Functor32x16x16 {
public:
  [[sycl::reqd_work_group_size(32, 16, 16)]] void operator()() const {}
};

template <int SIZE>
class FunctorTemp1D {
public:
  [[sycl::reqd_work_group_size(SIZE)]] void operator()() const {}
};
template <int SIZE, int SIZE1>
class FunctorTemp2D {
public:
  [[sycl::reqd_work_group_size(SIZE, SIZE1)]] void operator()() const {}
};
template <int SIZE, int SIZE1, int SIZE2>
class FunctorTemp3D {
public:
  [[sycl::reqd_work_group_size(SIZE, SIZE1, SIZE2)]] void operator()() const {}
};

int main() {
  q.submit([&](handler &h) {
    Functor32x16x16 f32x16x16;
    h.single_task<class kernel_name1>(f32x16x16);

    h.single_task<class kernel_name2>(
        []() [[sycl::reqd_work_group_size(8, 8, 8)]]{});

    FunctorTemp3D<2, 2, 2> ft3d;
    h.single_task<class kernel_name3>(ft3d);

    h.single_task<class kernel_name4>(
        []() [[sycl::reqd_work_group_size(1, 8, 2)]]{});

    Functor32x16 f32x16;
    h.single_task<class kernel_name5>(f32x16);

    h.single_task<class kernel_name6>(
        []() [[sycl::reqd_work_group_size(8, 8)]]{});

    FunctorTemp2D<2, 2> ft2d;
    h.single_task<class kernel_name7>(ft2d);

    h.single_task<class kernel_name8>(
        []() [[sycl::reqd_work_group_size(1, 8)]]{});

    Functor32 f32;
    h.single_task<class kernel_name9>(f32);

    h.single_task<class kernel_name10>(
        []() [[sycl::reqd_work_group_size(8)]]{});

    FunctorTemp1D<2> ft1d;
    h.single_task<class kernel_name11>(ft1d);

    h.single_task<class kernel_name12>(
        []() [[sycl::reqd_work_group_size(1)]]{});
  });
  return 0;
}

// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name1() #0 {{.*}} !reqd_work_group_size ![[WGSIZE3D32:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name2() #0 {{.*}} !reqd_work_group_size ![[WGSIZE3D88:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name3() #0 {{.*}} !reqd_work_group_size ![[WGSIZE3D22:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name4() #0 {{.*}} !reqd_work_group_size ![[WGSIZE3D2:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name5() #0 {{.*}} !reqd_work_group_size ![[WGSIZE2D32:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name6() #0 {{.*}} !reqd_work_group_size ![[WGSIZE2D88:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name7() #0 {{.*}} !reqd_work_group_size ![[WGSIZE2D22:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name8() #0 {{.*}} !reqd_work_group_size ![[WGSIZE2D2:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name9() #0 {{.*}} !reqd_work_group_size ![[WGSIZE1D32:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name10() #0 {{.*}} !reqd_work_group_size ![[WGSIZE1D8:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name11() #0 {{.*}} !reqd_work_group_size ![[WGSIZE1D22:[0-9]+]]
// CHECK: define {{.*}}spir_kernel void @{{.*}}kernel_name12() #0 {{.*}} !reqd_work_group_size ![[WGSIZE1D2:[0-9]+]]
// CHECK: ![[WGSIZE3D32]] = !{i32 16, i32 16, i32 32}
// CHECK: ![[WGSIZE3D88]] = !{i32 8, i32 8, i32 8}
// CHECK: ![[WGSIZE3D22]] = !{i32 2, i32 2, i32 2}
// CHECK: ![[WGSIZE3D2]] = !{i32 2, i32 8, i32 1}
// CHECK: ![[WGSIZE2D32]] = !{i32 16, i32 32}
// CHECK: ![[WGSIZE2D88]] = !{i32 8, i32 8}
// CHECK: ![[WGSIZE2D22]] = !{i32 2, i32 2}
// CHECK: ![[WGSIZE2D2]] = !{i32 8, i32 1}
// CHECK: ![[WGSIZE1D32]] = !{i32 32}
// CHECK: ![[WGSIZE1D8]] = !{i32 8}
// CHECK: ![[WGSIZE1D22]] = !{i32 2}
// CHECK: ![[WGSIZE1D2]] = !{i32 1}
