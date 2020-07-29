// RUN: %clangxx -fsycl -fsycl-explicit-simd -fsycl-device-only -O0 -S -emit-llvm -x c++ %s -o - | FileCheck %s
// This test checks that all SPIRV intrinsics are correctly
// translated into GenX counterparts (implemented in LowerCM.cpp)

#include <CL/sycl.hpp>
#include <CL/sycl/intel/esimd.hpp>

SYCL_EXTERNAL size_t __spirv_GlobalInvocationId_x();
SYCL_EXTERNAL size_t __spirv_GlobalInvocationId_y();
SYCL_EXTERNAL size_t __spirv_GlobalInvocationId_z();

SYCL_EXTERNAL size_t __spirv_GlobalSize_x();
SYCL_EXTERNAL size_t __spirv_GlobalSize_y();
SYCL_EXTERNAL size_t __spirv_GlobalSize_z();

SYCL_EXTERNAL size_t __spirv_GlobalOffset_x();
SYCL_EXTERNAL size_t __spirv_GlobalOffset_y();
SYCL_EXTERNAL size_t __spirv_GlobalOffset_z();

SYCL_EXTERNAL size_t __spirv_NumWorkgroups_x();
SYCL_EXTERNAL size_t __spirv_NumWorkgroups_y();
SYCL_EXTERNAL size_t __spirv_NumWorkgroups_z();

SYCL_EXTERNAL size_t __spirv_WorkgroupSize_x();
SYCL_EXTERNAL size_t __spirv_WorkgroupSize_y();
SYCL_EXTERNAL size_t __spirv_WorkgroupSize_z();

SYCL_EXTERNAL size_t __spirv_WorkgroupId_x();
SYCL_EXTERNAL size_t __spirv_WorkgroupId_y();
SYCL_EXTERNAL size_t __spirv_WorkgroupId_z();

SYCL_EXTERNAL size_t __spirv_LocalInvocationId_x();
SYCL_EXTERNAL size_t __spirv_LocalInvocationId_y();
SYCL_EXTERNAL size_t __spirv_LocalInvocationId_z();

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

size_t caller() {

  size_t DoNotOpt;
  cl::sycl::buffer<size_t, 1> buf(&DoNotOpt, 1);
  cl::sycl::queue().submit([&](cl::sycl::handler &cgh) {
    auto DoNotOptimize = buf.get_access<cl::sycl::access::mode::write>(cgh);

    kernel<class kernel_GlobalInvocationId_x>([=]() SYCL_ESIMD_KERNEL {
      *DoNotOptimize.get_pointer() = __spirv_GlobalInvocationId_x();
    });
    // CHECK-LABEL: @{{.*}}kernel_GlobalInvocationId_x
    // CHECK: [[CALL_ESIMD1:%.*]] = call <3 x i32> @llvm.genx.local.id.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD1]], i32 0
    // CHECK: [[CALL_ESIMD2:%.*]] = call <3 x i32> @llvm.genx.local.size.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD2]], i32 0
    // CHECK: {{.*}} call i32 @llvm.genx.group.id.x()

    kernel<class kernel_GlobalInvocationId_y>([=]() SYCL_ESIMD_KERNEL {
      *DoNotOptimize.get_pointer() = __spirv_GlobalInvocationId_y();
    });
    // CHECK-LABEL: @{{.*}}kernel_GlobalInvocationId_y
    // CHECK: [[CALL_ESIMD1:%.*]] = call <3 x i32> @llvm.genx.local.id.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD1]], i32 1
    // CHECK: [[CALL_ESIMD2:%.*]] = call <3 x i32> @llvm.genx.local.size.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD2]], i32 1
    // CHECK: {{.*}} call i32 @llvm.genx.group.id.y()

    kernel<class kernel_GlobalInvocationId_z>([=]() SYCL_ESIMD_KERNEL {
      *DoNotOptimize.get_pointer() = __spirv_GlobalInvocationId_z();
    });
    // CHECK-LABEL: @{{.*}}kernel_GlobalInvocationId_z
    // CHECK: [[CALL_ESIMD1:%.*]] = call <3 x i32> @llvm.genx.local.id.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD1]], i32 2
    // CHECK: [[CALL_ESIMD2:%.*]] = call <3 x i32> @llvm.genx.local.size.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD2]], i32 2
    // CHECK: {{.*}} call i32 @llvm.genx.group.id.z()

    kernel<class kernel_GlobalSize_x>([=]() SYCL_ESIMD_KERNEL {
      *DoNotOptimize.get_pointer() = __spirv_GlobalSize_x();
    });
    // CHECK-LABEL: @{{.*}}kernel_GlobalSize_x
    // CHECK: [[CALL_ESIMD1:%.*]] = call <3 x i32> @llvm.genx.local.size.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD1]], i32 0
    // CHECK: [[CALL_ESIMD2:%.*]] = call <3 x i32> @llvm.genx.group.count.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD2]], i32 0

    kernel<class kernel_GlobalSize_y>([=]() SYCL_ESIMD_KERNEL {
      *DoNotOptimize.get_pointer() = __spirv_GlobalSize_y();
    });
    // CHECK-LABEL: @{{.*}}kernel_GlobalSize_y
    // CHECK: [[CALL_ESIMD1:%.*]] = call <3 x i32> @llvm.genx.local.size.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD1]], i32 1
    // CHECK: [[CALL_ESIMD2:%.*]] = call <3 x i32> @llvm.genx.group.count.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD2]], i32 1

    kernel<class kernel_GlobalSize_z>([=]() SYCL_ESIMD_KERNEL {
      *DoNotOptimize.get_pointer() = __spirv_GlobalSize_z();
    });
    // CHECK-LABEL: @{{.*}}kernel_GlobalSize_z
    // CHECK: [[CALL_ESIMD1:%.*]] = call <3 x i32> @llvm.genx.local.size.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD1]], i32 2
    // CHECK: [[CALL_ESIMD2:%.*]] = call <3 x i32> @llvm.genx.group.count.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD2]], i32 2

    kernel<class kernel_GlobalOffset_x>([=]() SYCL_ESIMD_KERNEL {
      *DoNotOptimize.get_pointer() = __spirv_GlobalOffset_x();
    });
    // CHECK-LABEL: @{{.*}}kernel_GlobalOffset_x
    // CHECK: store i64 0

    kernel<class kernel_GlobalOffset_y>([=]() SYCL_ESIMD_KERNEL {
      *DoNotOptimize.get_pointer() = __spirv_GlobalOffset_y();
    });
    // CHECK-LABEL: @{{.*}}kernel_GlobalOffset_y
    // CHECK: store i64 0

    kernel<class kernel_GlobalOffset_z>([=]() SYCL_ESIMD_KERNEL {
      *DoNotOptimize.get_pointer() = __spirv_GlobalOffset_z();
    });
    // CHECK-LABEL: @{{.*}}kernel_GlobalOffset_z
    // CHECK: store i64 0

    kernel<class kernel_NumWorkgroups_x>([=]() SYCL_ESIMD_KERNEL {
      *DoNotOptimize.get_pointer() = __spirv_NumWorkgroups_x();
    });
    // CHECK-LABEL: @{{.*}}kernel_NumWorkgroups_x
    // CHECK: [[CALL_ESIMD:%.*]] = call <3 x i32> @llvm.genx.group.count.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD]], i32 0

    kernel<class kernel_NumWorkgroups_y>([=]() SYCL_ESIMD_KERNEL {
      *DoNotOptimize.get_pointer() = __spirv_NumWorkgroups_y();
    });
    // CHECK-LABEL: @{{.*}}kernel_NumWorkgroups_y
    // CHECK: [[CALL_ESIMD:%.*]] = call <3 x i32> @llvm.genx.group.count.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD]], i32 1

    kernel<class kernel_NumWorkgroups_z>([=]() SYCL_ESIMD_KERNEL {
      *DoNotOptimize.get_pointer() = __spirv_NumWorkgroups_z();
    });
    // CHECK-LABEL: @{{.*}}kernel_NumWorkgroups_z
    // CHECK: [[CALL_ESIMD:%.*]] = call <3 x i32> @llvm.genx.group.count.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD]], i32 2

    kernel<class kernel_WorkgroupSize_x>([=]() SYCL_ESIMD_KERNEL {
      *DoNotOptimize.get_pointer() = __spirv_WorkgroupSize_x();
    });
    // CHECK-LABEL: @{{.*}}kernel_WorkgroupSize_x
    // CHECK: [[CALL_ESIMD:%.*]] = call <3 x i32> @llvm.genx.local.size.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD]], i32 0

    kernel<class kernel_WorkgroupSize_y>([=]() SYCL_ESIMD_KERNEL {
      *DoNotOptimize.get_pointer() = __spirv_WorkgroupSize_y();
    });
    // CHECK-LABEL: @{{.*}}kernel_WorkgroupSize_y
    // CHECK: [[CALL_ESIMD:%.*]] = call <3 x i32> @llvm.genx.local.size.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD]], i32 1

    kernel<class kernel_WorkgroupSize_z>([=]() SYCL_ESIMD_KERNEL {
      *DoNotOptimize.get_pointer() = __spirv_WorkgroupSize_z();
    });
    // CHECK-LABEL: @{{.*}}kernel_WorkgroupSize_z
    // CHECK: [[CALL_ESIMD:%.*]] = call <3 x i32> @llvm.genx.local.size.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD]], i32 2

    kernel<class kernel_WorkgroupId_x>([=]() SYCL_ESIMD_KERNEL {
      *DoNotOptimize.get_pointer() = __spirv_WorkgroupId_x();
    });
    // CHECK-LABEL: @{{.*}}kernel_WorkgroupId_x
    // CHECK: {{.*}} call i32 @llvm.genx.group.id.x()

    kernel<class kernel_WorkgroupId_y>([=]() SYCL_ESIMD_KERNEL {
      *DoNotOptimize.get_pointer() = __spirv_WorkgroupId_y();
    });
    // CHECK-LABEL: @{{.*}}kernel_WorkgroupId_y
    // CHECK: {{.*}} call i32 @llvm.genx.group.id.y()

    kernel<class kernel_WorkgroupId_z>([=]() SYCL_ESIMD_KERNEL {
      *DoNotOptimize.get_pointer() = __spirv_WorkgroupId_z();
    });
    // CHECK-LABEL: @{{.*}}kernel_WorkgroupId_z
    // CHECK: {{.*}} call i32 @llvm.genx.group.id.z()

    kernel<class kernel_LocalInvocationId_x>([=]() SYCL_ESIMD_KERNEL {
      *DoNotOptimize.get_pointer() = __spirv_LocalInvocationId_x();
    });
    // CHECK-LABEL: @{{.*}}kernel_LocalInvocationId_x
    // CHECK: [[CALL_ESIMD:%.*]] = call <3 x i32> @llvm.genx.local.id.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD]], i32 0

    kernel<class kernel_LocalInvocationId_y>([=]() SYCL_ESIMD_KERNEL {
      *DoNotOptimize.get_pointer() = __spirv_LocalInvocationId_y();
    });
    // CHECK-LABEL: @{{.*}}kernel_LocalInvocationId_y
    // CHECK: [[CALL_ESIMD:%.*]] = call <3 x i32> @llvm.genx.local.id.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD]], i32 1

    kernel<class kernel_LocalInvocationId_z>([=]() SYCL_ESIMD_KERNEL {
      *DoNotOptimize.get_pointer() = __spirv_LocalInvocationId_z();
    });
    // CHECK-LABEL: @{{.*}}kernel_LocalInvocationId_z
    // CHECK: [[CALL_ESIMD:%.*]] = call <3 x i32> @llvm.genx.local.id.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD]], i32 2
  });
  return DoNotOpt;
}
