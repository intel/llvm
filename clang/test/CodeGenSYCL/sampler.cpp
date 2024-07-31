// RUN: %clang_cc1 -fno-sycl-force-inline-kernel-lambda -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck --enable-var-scope %s
// CHECK: define {{.*}}spir_kernel void @{{[a-zA-Z0-9_]+}}(target("spirv.Sampler") [[SAMPLER_ARG:%[a-zA-Z0-9_]+]])
// CHECK-NEXT: entry:
// CHECK-NEXT: [[SAMPLER_ARG]].addr = alloca target("spirv.Sampler"), align 8
// CHECK: [[ANON:%[a-zA-Z0-9_]+]] = alloca %class.anon, align 8
// CHECK: [[ANONCAST:%[a-zA-Z0-9_.]+]] = addrspacecast ptr [[ANON]] to ptr addrspace(4)
// CHECK: store target("spirv.Sampler") [[SAMPLER_ARG]], ptr addrspace(4) [[SAMPLER_ARG]].addr.ascast, align 8
// CHECK-NEXT: [[GEP:%[a-zA-z0-9]+]]  = getelementptr inbounds %class.anon, ptr addrspace(4) [[ANONCAST]], i32 0, i32 0
// CHECK-NEXT: [[GEP2:%[a-zA-z0-9]+]]  = getelementptr inbounds %class.anon, ptr addrspace(4) [[ANONCAST]], i32 0, i32 0
// CHECK-NEXT: [[LOAD_SAMPLER_ARG:%[0-9]+]] = load target("spirv.Sampler"), ptr addrspace(4) [[SAMPLER_ARG]].addr.ascast, align 8
// CHECK-NEXT: call spir_func void @{{[a-zA-Z0-9_]+}}(ptr addrspace(4) {{[^,]*}} [[GEP2]], target("spirv.Sampler") [[LOAD_SAMPLER_ARG]])
//

// CHECK: define {{.*}}spir_kernel void @{{[a-zA-Z0-9_]+}}(target("spirv.Sampler") [[SAMPLER_ARG_WRAPPED:%[a-zA-Z0-9_]+]], i32 noundef [[ARG_A:%[a-zA-Z0-9_]+]])

// Check alloca
// CHECK: [[SAMPLER_ARG_WRAPPED]].addr = alloca target("spirv.Sampler"), align 8
// CHECK: [[ARG_A]].addr = alloca i32, align 4
// CHECK: [[LAMBDAA:%[a-zA-Z0-9_]+]] = alloca %class.anon.0, align 8
// CHECK: [[LAMBDA:%[a-zA-Z0-9_.]+]] = addrspacecast ptr [[LAMBDAA]] to ptr addrspace(4)

// Check argument store
// CHECK: store target("spirv.Sampler") [[SAMPLER_ARG_WRAPPED]], ptr addrspace(4) [[SAMPLER_ARG_WRAPPED]].addr.ascast, align 8
// CHECK: store i32 [[ARG_A]], ptr addrspace(4) [[ARG_A]].addr.ascast, align 4

// Initialize 'a'
// CHECK: [[GEP_LAMBDA:%[a-zA-z0-9]+]] = getelementptr inbounds %class.anon.0, ptr addrspace(4) [[LAMBDA]], i32 0, i32 0
// CHECK: [[GEP_A:%[a-zA-Z0-9]+]] = getelementptr inbounds %struct.sampler_wrapper, ptr addrspace(4) [[GEP_LAMBDA]], i32 0, i32 1
// CHECK: [[LOAD_A:%[0-9]+]] = load i32, ptr addrspace(4) [[ARG_A]].addr.ascast, align 4
// CHECK: store i32 [[LOAD_A]], ptr addrspace(4) [[GEP_A]], align 8

// Initialize wrapped sampler 'smpl'
// CHECK: [[GEP_LAMBDA_0:%[a-zA-z0-9]+]] = getelementptr inbounds %class.anon.0, ptr addrspace(4) [[LAMBDA]], i32 0, i32 0
// CHECK: [[GEP_SMPL:%[a-zA-Z0-9]+]] = getelementptr inbounds %struct.sampler_wrapper, ptr addrspace(4) [[GEP_LAMBDA_0]], i32 0, i32 0
// CHECK: [[LOAD_SMPL:%[0-9]+]] = load target("spirv.Sampler"), ptr addrspace(4) [[SAMPLER_ARG_WRAPPED]].addr.ascast, align 8
// CHECK: call spir_func void @{{[a-zA-Z0-9_]+}}(ptr addrspace(4) {{.*}}, target("spirv.Sampler") [[LOAD_SMPL]])
//
#include "Inputs/sycl.hpp"

struct sampler_wrapper {
  sycl::sampler smpl;
  int a;
};

template <typename KernelName, typename KernelType>
__attribute__((sycl_kernel)) void kernel_single_task(const KernelType &kernelFunc) {
  kernelFunc();
}

int main() {
  sycl::sampler smplr;
  kernel_single_task<class first_kernel>([=]() {
    smplr.use();
  });

  sampler_wrapper wrappedSampler = {smplr, 1};
  kernel_single_task<class second_kernel>([=]() {
    wrappedSampler.smpl.use();
  });

  return 0;
}
