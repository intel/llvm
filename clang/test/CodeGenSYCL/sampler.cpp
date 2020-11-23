// RUN: %clang_cc1 -fsycl -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -o - | FileCheck --enable-var-scope %s
// CHECK: define spir_kernel void @{{[a-zA-Z0-9_]+}}(%opencl.sampler_t addrspace(2)* [[SAMPLER_ARG:%[a-zA-Z0-9_]+]])
// CHECK-NEXT: entry:
// CHECK-NEXT: [[SAMPLER_ARG]].addr = alloca %opencl.sampler_t addrspace(2)*, align 8
// CHECK-NEXT: [[ANON:%[0-9]+]] = alloca %"class.{{.*}}.anon", align 8
// CHECK-NEXT: store %opencl.sampler_t addrspace(2)* [[SAMPLER_ARG]], %opencl.sampler_t addrspace(2)** [[SAMPLER_ARG]].addr, align 8
// CHECK-NEXT: [[BITCAST:%[0-9]+]] = bitcast %"class.{{.*}}.anon"* [[ANON]] to i8*
// CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* [[BITCAST]]) #4
// CHECK-NEXT: [[GEP:%[0-9]+]]  = getelementptr inbounds %"class.{{.*}}.anon", %"class.{{.*}}.anon"* [[ANON]], i32 0, i32 0
// CHECK-NEXT: [[LOAD_SAMPLER_ARG:%[0-9]+]] = load %opencl.sampler_t addrspace(2)*, %opencl.sampler_t addrspace(2)** [[SAMPLER_ARG]].addr, align 8
// CHECK-NEXT: [[GEPCAST:%[0-9]+]] = addrspacecast %"class{{.*}}.cl::sycl::sampler"* [[GEP]] to %"class{{.*}}.cl::sycl::sampler" addrspace(4)*
// CHECK-NEXT: call spir_func void @{{[a-zA-Z0-9_]+}}(%"class.{{.*}}.cl::sycl::sampler" addrspace(4)* {{[^,]*}} [[GEPCAST]], %opencl.sampler_t addrspace(2)* [[LOAD_SAMPLER_ARG]])
//

// CHECK: define spir_kernel void @{{[a-zA-Z0-9_]+}}(%opencl.sampler_t addrspace(2)* [[SAMPLER_ARG_WRAPPED:%[a-zA-Z0-9_]+]], i32 [[ARG_A:%[a-zA-Z0-9_]+]])

// Check alloca
// CHECK: [[SAMPLER_ARG_WRAPPED]].addr = alloca %opencl.sampler_t addrspace(2)*, align 8
// CHECK: [[ARG_A]].addr = alloca i32, align 4
// CHECK: [[LAMBDA:%[0-9]+]] = alloca %"class.{{.*}}.anon.0", align 8

// Check argument store
// CHECK: store %opencl.sampler_t addrspace(2)* [[SAMPLER_ARG_WRAPPED]], %opencl.sampler_t addrspace(2)** [[SAMPLER_ARG_WRAPPED]].addr, align 8
// CHECK: store i32 [[ARG_A]], i32* [[ARG_A]].addr, align 4

// Initialize 'a'
// CHECK: [[GEP_LAMBDA:%[0-9]+]] = getelementptr inbounds %"class.{{.*}}.anon.0", %"class.{{.*}}.anon.0"* [[LAMBDA]], i32 0, i32 0
// CHECK: [[GEP_A:%[a-zA-Z0-9]+]] = getelementptr inbounds %struct.{{.*}}.sampler_wrapper, %struct.{{.*}}.sampler_wrapper* [[GEP_LAMBDA]], i32 0, i32 1
// CHECK: [[LOAD_A:%[0-9]+]] = load i32, i32* [[ARG_A]].addr, align 4
// CHECK: store i32 [[LOAD_A]], i32* [[GEP_A]], align 8

// Initialize wrapped sampler 'smpl'
// CHECK: [[GEP_LAMBDA_0:%[0-9]+]] = getelementptr inbounds %"class.{{.*}}.anon.0", %"class.{{.*}}.anon.0"* %0, i32 0, i32 0
// CHECK: [[GEP_SMPL:%[a-zA-Z0-9]+]] = getelementptr inbounds %struct.{{.*}}.sampler_wrapper, %struct.{{.*}}.sampler_wrapper* [[GEP_LAMBDA_0]], i32 0, i32 0
// CHECK: [[LOAD_SMPL:%[0-9]+]] = load %opencl.sampler_t addrspace(2)*, %opencl.sampler_t addrspace(2)** [[SAMPLER_ARG_WRAPPED]].addr, align 8
// CHECK: call spir_func void @{{[a-zA-Z0-9_]+}}(%"class.{{.*}}.cl::sycl::sampler" addrspace(4)* {{.*}}, %opencl.sampler_t addrspace(2)* [[LOAD_SMPL]])
//
#include "Inputs/sycl.hpp"

struct sampler_wrapper {
  cl::sycl::sampler smpl;
  int a;
};

template <typename KernelName, typename KernelType>
__attribute__((sycl_kernel)) void kernel_single_task(const KernelType &kernelFunc) {
  kernelFunc();
}

int main() {
  cl::sycl::sampler smplr;
  kernel_single_task<class first_kernel>([=]() {
    smplr.use();
  });

  sampler_wrapper wrappedSampler = {smplr, 1};
  kernel_single_task<class second_kernel>([=]() {
    wrappedSampler.smpl.use();
  });

  return 0;
}
