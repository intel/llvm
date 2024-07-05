// RUN: %clang_cc1 -fno-sycl-force-inline-kernel-lambda -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -disable-llvm-passes -sycl-std=2020 -emit-llvm -o - %s | FileCheck %s

// RUN: %clang_cc1 -fno-sycl-force-inline-kernel-lambda -fsycl-is-device -internal-isystem %S/Inputs -triple amdgcn-amd-amdhsa -disable-llvm-passes -sycl-std=2020 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -fno-sycl-force-inline-kernel-lambda -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx-nvidia-cuda -disable-llvm-passes -sycl-std=2020 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -fno-sycl-force-inline-kernel-lambda -fsycl-is-device -internal-isystem %S/Inputs -triple nvptx64-nvidia-cuda -disable-llvm-passes -sycl-std=2020 -emit-llvm -o - %s | FileCheck %s

// Tests that work_group_size_hint and reqd_work_group_size generate the same
// metadata nodes for the same arguments.

#include "sycl.hpp"

using namespace sycl;

int main() {
  queue q;

  q.submit([&](handler &h) {
    // CHECK: define {{.*}} void @{{.*}}kernel_1d() #0 {{.*}} !work_group_size_hint ![[WGSH1D:[0-9]+]]{{.*}} !work_group_num_dim ![[NDRWGS1D:[0-9]+]]{{.*}} !reqd_work_group_size ![[WGSH1D]]
    h.single_task<class kernel_1d>([]() [[sycl::work_group_size_hint(8)]] [[sycl::reqd_work_group_size(8)]] {});
  });

  q.submit([&](handler &h) {
    // CHECK: define {{.*}} void @{{.*}}kernel_2d() #0 {{.*}} !work_group_size_hint ![[WGSH2D:[0-9]+]]{{.*}} !work_group_num_dim ![[NDRWGS2D:[0-9]+]]{{.*}} !reqd_work_group_size ![[WGSH2D:[0-9]+]]{{.*}}
    h.single_task<class kernel_2d>([]() [[sycl::work_group_size_hint(8, 16)]] [[sycl::reqd_work_group_size(8, 16)]] {});
  });

  q.submit([&](handler &h) {
    // CHECK: define {{.*}} void @{{.*}}kernel_3d() #0 {{.*}} !work_group_size_hint ![[WG3D:[0-9]+]]{{.*}} !work_group_num_dim ![[NDRWGS3D:[0-9]+]]{{.*}} !reqd_work_group_size ![[WG3D]]
    h.single_task<class kernel_3d>([]() [[sycl::work_group_size_hint(8, 16, 32)]] [[sycl::reqd_work_group_size(8, 16, 32)]] {});
  });
}

// CHECK: ![[WGSH1D]] = !{i32 8, i32 1, i32 1}
// CHECK: ![[NDRWGS1D]] = !{i32 1}
// CHECK: ![[WGSH2D]] = !{i32 16, i32 8, i32 1}
// CHECK: ![[NDRWGS2D]] = !{i32 2}
// CHECK: ![[WG3D]] = !{i32 32, i32 16, i32 8}
// CHECK: ![[NDRWGS3D]] = !{i32 3}
