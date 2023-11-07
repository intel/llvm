// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -disable-llvm-passes -triple amdgcn-amd-amdhsa -target-cpu gfx1010 -S -emit-llvm -o - %s | FileCheck -check-prefix=CHECK_AMD_32 %s

// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -disable-llvm-passes -triple amdgcn-amd-amdhsa -target-cpu gfx90a -S -emit-llvm -o - %s | FileCheck -check-prefix=CHECK_AMD_64 %s

// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -disable-llvm-passes -triple nvptx-unknown-unknown -target-cpu sm_90 -S -emit-llvm -o - %s | FileCheck -check-prefix=CHECK_CUDA_32 %s

// Check that incorrect values specified for reqd_sub_group_size are ignored.
// CDNA supports only 64 wave front size, for those GPUs allow subgroup size of
// 64. Some GPUs support both 32 and 64, for those (and the rest) only allow
// 32. For CUDA only allow 32.

#include "sycl.hpp"

int main() {

  sycl::queue Q;

  Q.submit([&](sycl::handler &h) {
    h.single_task<class Kernel_1>([=] [[sycl::reqd_sub_group_size(64)]] {});
  });

  Q.submit([&](sycl::handler &h) {
    h.single_task<class Kernel_2>([=] [[sycl::reqd_sub_group_size(32)]] {});
  });

  Q.submit([&](sycl::handler &h) {
    h.single_task<class Kernel_3>([=] [[sycl::reqd_sub_group_size(8)]] {});
  });

  return 0;
}

// CHECK_AMD_32: define {{.*}}amdgpu_kernel void @{{.*}}Kernel_1() #0 {{.*}}
// CHECK_AMD_32-NOT: intel_reqd_sub_group_size
// CHECK_AMD_32: define {{.*}}amdgpu_kernel void @{{.*}}Kernel_2() #0 {{.*}} !intel_reqd_sub_group_size ![[IRSGS_32:[0-9]+]]
// CHECK_AMD_32: define {{.*}}amdgpu_kernel void @{{.*}}Kernel_3() #0 {{.*}}
// CHECK_AMD_32-NOT: intel_reqd_sub_group_size
// CHECK_AMD_32: ![[IRSGS_32]] = !{i32 32}

// CHECK_AMD_64: define {{.*}}amdgpu_kernel void @{{.*}}Kernel_1() #0 {{.*}} !intel_reqd_sub_group_size ![[IRSGS_64:[0-9]+]]
// CHECK_AMD_64: define {{.*}}amdgpu_kernel void @{{.*}}Kernel_2() #0 {{.*}}
// CHECK_AMD_64-NOT: intel_reqd_sub_group_size
// CHECK_AMD_64: define {{.*}}amdgpu_kernel void @{{.*}}Kernel_3() #0 {{.*}}
// CHECK_AMD_64-NOT: intel_reqd_sub_group_size
// CHECK_AMD_64: ![[IRSGS_64]] = !{i32 64}

// CHECK_CUDA_32: define {{.*}} void @{{.*}}Kernel_1() #0 {{.*}}
// CHECK_CUDA_32-NOT: intel_reqd_sub_group_size
// CHECK_CUDA_32: define {{.*}} void @{{.*}}Kernel_2() #0 {{.*}} !intel_reqd_sub_group_size ![[IRSGS_32:[0-9]+]]
// CHECK_CUDA_32: define {{.*}} void @{{.*}}Kernel_3() #0 {{.*}}
// CHECK_CUDA_32-NOT: intel_reqd_sub_group_size
// CHECK_CUDA_32: ![[IRSGS_32]] = !{i32 32}
