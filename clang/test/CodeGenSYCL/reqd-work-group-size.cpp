// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple amdgcn-amd-amdhsa -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx-nvidia-cuda -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-NVPTX
// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-NVPTX

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
class CLFunctor32x16x16 {
public:
  [[cl::reqd_work_group_size(32, 16, 16)]] void operator()() const {}
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
template <int SIZE, int SIZE1, int SIZE2>
class CLFunctorTemp3D {
public:
  [[cl::reqd_work_group_size(SIZE, SIZE1, SIZE2)]] void operator()() const {}
};

int main() {
  q.submit([&](handler &h) {
    CLFunctor32x16x16 clf32x16x16;
    h.single_task<class kernel_name1>(clf32x16x16);

    
    h.single_task<class kernel_name3>(
        []() [[cl::reqd_work_group_size(8, 8, 8)]]{});

    CLFunctorTemp3D<2, 2, 2> clft3d;
    h.single_task<class kernel_name4>(clft3d);

    h.single_task<class kernel_name6>(
        []() [[cl::reqd_work_group_size(1, 8, 2)]]{});

    Functor32x16x16 f32x16x16;
    h.single_task<class kernel_name7>(f32x16x16);


    h.single_task<class kernel_name9>(
        []() [[sycl::reqd_work_group_size(8, 8, 8)]]{});

    FunctorTemp3D<2, 2, 2> ft3d;
    h.single_task<class kernel_name10>(ft3d);


    h.single_task<class kernel_name12>(
        []() [[sycl::reqd_work_group_size(1, 8, 2)]]{});

    Functor32x16 f32x16;
    h.single_task<class kernel_name13>(f32x16);


    h.single_task<class kernel_name15>(
        []() [[sycl::reqd_work_group_size(8, 8)]]{});

    FunctorTemp2D<2, 2> ft2d;
    h.single_task<class kernel_name16>(ft2d);


    h.single_task<class kernel_name18>(
        []() [[sycl::reqd_work_group_size(1, 8)]]{});

    Functor32 f32;
    h.single_task<class kernel_name19>(f32);


    h.single_task<class kernel_name21>(
        []() [[sycl::reqd_work_group_size(8)]]{});

    FunctorTemp1D<2> ft1d;
    h.single_task<class kernel_name22>(ft1d);

    h.single_task<class kernel_name24>(
        []() [[sycl::reqd_work_group_size(1)]]{});
  });
  return 0;
}

// CHECK: define {{.*}} void @{{.*}}kernel_name1() #0 {{.*}} !work_group_num_dim ![[NDRWGS3D:[0-9]+]] !reqd_work_group_size ![[WGSIZE3D32:[0-9]+]]
// CHECK: define {{.*}} void @{{.*}}kernel_name3() #0 {{.*}} !work_group_num_dim ![[NDRWGS3D:[0-9]+]] !reqd_work_group_size ![[WGSIZE3D88:[0-9]+]]
// CHECK: define {{.*}} void @{{.*}}kernel_name4() #0 {{.*}} !work_group_num_dim ![[NDRWGS3D:[0-9]+]] !reqd_work_group_size ![[WGSIZE3D22:[0-9]+]]
// CHECK: define {{.*}} void @{{.*}}kernel_name6() #0 {{.*}} !work_group_num_dim ![[NDRWGS3D:[0-9]+]] !reqd_work_group_size ![[WGSIZE3D2:[0-9]+]]
// CHECK: define {{.*}} void @{{.*}}kernel_name7() #0 {{.*}} !work_group_num_dim ![[NDRWGS3D:[0-9]+]] !reqd_work_group_size ![[WGSIZE3D32]]
// CHECK: define {{.*}} void @{{.*}}kernel_name9() #0 {{.*}} !work_group_num_dim ![[NDRWGS3D:[0-9]+]] !reqd_work_group_size ![[WGSIZE3D88]]
// CHECK: define {{.*}} void @{{.*}}kernel_name10() #0 {{.*}} !work_group_num_dim ![[NDRWGS3D:[0-9]+]] !reqd_work_group_size ![[WGSIZE3D22]]
// CHECK: define {{.*}} void @{{.*}}kernel_name12() #0 {{.*}} !work_group_num_dim ![[NDRWGS3D:[0-9]+]] !reqd_work_group_size ![[WGSIZE3D2]]
// CHECK: define {{.*}} void @{{.*}}kernel_name13() #0 {{.*}} !work_group_num_dim ![[NDRWGS2D:[0-9]+]] !reqd_work_group_size ![[WGSIZE2D32:[0-9]+]]
// CHECK: define {{.*}} void @{{.*}}kernel_name15() #0 {{.*}} !work_group_num_dim ![[NDRWGS2D:[0-9]+]] !reqd_work_group_size ![[WGSIZE2D88:[0-9]+]]
// CHECK: define {{.*}} void @{{.*}}kernel_name16() #0 {{.*}} !work_group_num_dim ![[NDRWGS2D:[0-9]+]] !reqd_work_group_size ![[WGSIZE2D22:[0-9]+]]
// CHECK: define {{.*}} void @{{.*}}kernel_name18() #0 {{.*}} !work_group_num_dim ![[NDRWGS2D:[0-9]+]] !reqd_work_group_size ![[WGSIZE2D2_or_WGSIZE1D8:[0-9]+]]
// CHECK: define {{.*}} void @{{.*}}kernel_name19() #0 {{.*}} !work_group_num_dim ![[NDRWGS1D:[0-9]+]] !reqd_work_group_size ![[WGSIZE1D32:[0-9]+]]
// CHECK: define {{.*}} void @{{.*}}kernel_name21() #0 {{.*}} !work_group_num_dim ![[NDRWGS1D:[0-9]+]] !reqd_work_group_size ![[WGSIZE2D2_or_WGSIZE1D8]]
// CHECK: define {{.*}} void @{{.*}}kernel_name22() #0 {{.*}} !work_group_num_dim ![[NDRWGS1D:[0-9]+]] !reqd_work_group_size ![[WGSIZE1D22:[0-9]+]]
// CHECK: define {{.*}} void @{{.*}}kernel_name24() #0 {{.*}} !work_group_num_dim ![[NDRWGS1D:[0-9]+]] !reqd_work_group_size ![[WGSIZE1D2:[0-9]+]]

// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name1, !"reqntidx", i32 16}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name1, !"reqntidy", i32 16}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name1, !"reqntidz", i32 32}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name3, !"reqntidx", i32 8}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name3, !"reqntidy", i32 8}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name3, !"reqntidz", i32 8}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name4, !"reqntidx", i32 2}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name4, !"reqntidy", i32 2}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name4, !"reqntidz", i32 2}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name6, !"reqntidx", i32 2}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name6, !"reqntidy", i32 8}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name6, !"reqntidz", i32 1}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name7, !"reqntidx", i32 16}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name7, !"reqntidy", i32 16}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name7, !"reqntidz", i32 32}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name9, !"reqntidx", i32 8}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name9, !"reqntidy", i32 8}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name9, !"reqntidz", i32 8}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name10, !"reqntidx", i32 2}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name10, !"reqntidy", i32 2}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name10, !"reqntidz", i32 2}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name12, !"reqntidx", i32 2}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name12, !"reqntidy", i32 8}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name12, !"reqntidz", i32 1}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name13, !"reqntidx", i32 16}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name13, !"reqntidy", i32 32}
// CHECK-NVPTX-NOT: = !{ptr @{{.*}}kernel_name13, !"reqntidz"
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name15, !"reqntidx", i32 8}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name15, !"reqntidy", i32 8}
// CHECK-NVPTX-NOT: = !{ptr @{{.*}}kernel_name15, !"reqntidz"
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name16, !"reqntidx", i32 2}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name16, !"reqntidy", i32 2}
// CHECK-NVPTX-NOT: = !{ptr @{{.*}}kernel_name16, !"reqntidz"
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name18, !"reqntidx", i32 8}
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name18, !"reqntidy", i32 1}
// CHECK-NVPTX-NOT: = !{ptr @{{.*}}kernel_name18, !"reqntidz"
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name19, !"reqntidx", i32 32}
// CHECK-NVPTX-NOT: = !{ptr @{{.*}}kernel_name19, !"reqntidy",
// CHECK-NVPTX-NOT: = !{ptr @{{.*}}kernel_name19, !"reqntidz",
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name21, !"reqntidx", i32 8}
// CHECK-NVPTX-NOT: = !{ptr @{{.*}}kernel_name21, !"reqntidy",
// CHECK-NVPTX-NOT: = !{ptr @{{.*}}kernel_name21, !"reqntidz",
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name22, !"reqntidx", i32 2}
// CHECK-NVPTX-NOT: = !{ptr @{{.*}}kernel_name22, !"reqntidy",
// CHECK-NVPTX-NOT: = !{ptr @{{.*}}kernel_name22, !"reqntidz",
// CHECK-NVPTX: = !{ptr @{{.*}}kernel_name24, !"reqntidx", i32 1}
// CHECK-NVPTX-NOT: = !{ptr @{{.*}}kernel_name24, !"reqntidy",
// CHECK-NVPTX-NOT: = !{ptr @{{.*}}kernel_name24, !"reqntidz",

// CHECK: ![[NDRWGS3D]] = !{i32 3}
// CHECK: ![[WGSIZE3D32]] = !{i32 16, i32 16, i32 32}
// CHECK: ![[WGSIZE3D88]] = !{i32 8, i32 8, i32 8}
// CHECK: ![[WGSIZE3D22]] = !{i32 2, i32 2, i32 2}
// CHECK: ![[WGSIZE3D2]] = !{i32 2, i32 8, i32 1}
// CHECK: ![[NDRWGS2D]] = !{i32 2}
// CHECK: ![[WGSIZE2D32]] = !{i32 16, i32 32, i32 1}
// CHECK: ![[WGSIZE2D88]] = !{i32 8, i32 8, i32 1}
// CHECK: ![[WGSIZE2D22]] = !{i32 2, i32 2, i32 1}
// CHECK: ![[WGSIZE2D2_or_WGSIZE1D8]] = !{i32 8, i32 1, i32 1}
// CHECK: ![[NDRWGS1D]] = !{i32 1}
// CHECK: ![[WGSIZE1D32]] = !{i32 32, i32 1, i32 1}
// CHECK: ![[WGSIZE1D22]] = !{i32 2, i32 1, i32 1}
// CHECK: ![[WGSIZE1D2]] = !{i32 1, i32 1, i32 1}
