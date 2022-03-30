// RUN: %clangxx -fsycl -fsycl-device-only -flegacy-pass-manager -S -emit-llvm -x c++ %s -o %t-lgcy
// RUN: sycl-post-link -split-esimd -lower-esimd -O0 -S %t-lgcy -o %t-lgcy.table
// RUN: FileCheck %s -input-file=%t-lgcy_esimd_0.ll

// RUN: %clangxx -fsycl -fsycl-device-only -fno-legacy-pass-manager -S -emit-llvm -x c++ %s -o %t
// RUN: sycl-post-link -split-esimd -lower-esimd -O0 -S %t -o %t.table
// RUN: FileCheck %s -input-file=%t_esimd_0.ll

// This test checks that all LLVM-IR instructions that work with SPIR-V builtins
// are correctly translated into GenX counterparts (implemented in
// LowerESIMD.cpp)

#include <CL/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

size_t caller() {

  size_t DoNotOpt[1];
  uint32_t DoNotOpt32[1];
  size_t DoNotOptXYZ[3];

  cl::sycl::queue().submit([&](cl::sycl::handler &cgh) {
    auto DoNotOptimize = &DoNotOpt[0];
    auto DoNotOptimize32 = &DoNotOpt32[0];

    kernel<class kernel_GlobalInvocationId_x>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimize[0] = __spirv_GlobalInvocationId_x();
    });
    // CHECK-LABEL: @{{.*}}kernel_GlobalInvocationId_x
    // CHECK: [[CALL_ESIMD1:%.*]] = call <3 x i32> @llvm.genx.local.id.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD1]], i32 0
    // CHECK: [[CALL_ESIMD2:%.*]] = call <3 x i32> @llvm.genx.local.size.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD2]], i32 0
    // CHECK: {{.*}} call i32 @llvm.genx.group.id.x()

    kernel<class kernel_GlobalInvocationId_y>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimize[0] = __spirv_GlobalInvocationId_y();
    });
    // CHECK-LABEL: @{{.*}}kernel_GlobalInvocationId_y
    // CHECK: [[CALL_ESIMD1:%.*]] = call <3 x i32> @llvm.genx.local.id.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD1]], i32 1
    // CHECK: [[CALL_ESIMD2:%.*]] = call <3 x i32> @llvm.genx.local.size.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD2]], i32 1
    // CHECK: {{.*}} call i32 @llvm.genx.group.id.y()

    kernel<class kernel_GlobalInvocationId_z>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimize[0] = __spirv_GlobalInvocationId_z();
    });
    // CHECK-LABEL: @{{.*}}kernel_GlobalInvocationId_z
    // CHECK: [[CALL_ESIMD1:%.*]] = call <3 x i32> @llvm.genx.local.id.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD1]], i32 2
    // CHECK: [[CALL_ESIMD2:%.*]] = call <3 x i32> @llvm.genx.local.size.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD2]], i32 2
    // CHECK: {{.*}} call i32 @llvm.genx.group.id.z()

    kernel<class kernel_GlobalSize_x>(
        [=]() SYCL_ESIMD_KERNEL { DoNotOptimize[0] = __spirv_GlobalSize_x(); });
    // CHECK-LABEL: @{{.*}}kernel_GlobalSize_x
    // CHECK: [[CALL_ESIMD1:%.*]] = call <3 x i32> @llvm.genx.local.size.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD1]], i32 0
    // CHECK: [[CALL_ESIMD2:%.*]] = call <3 x i32> @llvm.genx.group.count.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD2]], i32 0

    kernel<class kernel_GlobalSize_y>(
        [=]() SYCL_ESIMD_KERNEL { DoNotOptimize[0] = __spirv_GlobalSize_y(); });
    // CHECK-LABEL: @{{.*}}kernel_GlobalSize_y
    // CHECK: [[CALL_ESIMD1:%.*]] = call <3 x i32> @llvm.genx.local.size.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD1]], i32 1
    // CHECK: [[CALL_ESIMD2:%.*]] = call <3 x i32> @llvm.genx.group.count.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD2]], i32 1

    kernel<class kernel_GlobalSize_z>(
        [=]() SYCL_ESIMD_KERNEL { DoNotOptimize[0] = __spirv_GlobalSize_z(); });
    // CHECK-LABEL: @{{.*}}kernel_GlobalSize_z
    // CHECK: [[CALL_ESIMD1:%.*]] = call <3 x i32> @llvm.genx.local.size.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD1]], i32 2
    // CHECK: [[CALL_ESIMD2:%.*]] = call <3 x i32> @llvm.genx.group.count.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD2]], i32 2

    kernel<class kernel_GlobalOffset_x>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimize[0] = __spirv_GlobalOffset_x();
    });
    // CHECK-LABEL: @{{.*}}kernel_GlobalOffset_x
    // CHECK: store i64 0

    kernel<class kernel_GlobalOffset_y>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimize[0] = __spirv_GlobalOffset_y();
    });
    // CHECK-LABEL: @{{.*}}kernel_GlobalOffset_y
    // CHECK: store i64 0

    kernel<class kernel_GlobalOffset_z>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimize[0] = __spirv_GlobalOffset_z();
    });
    // CHECK-LABEL: @{{.*}}kernel_GlobalOffset_z
    // CHECK: store i64 0

    kernel<class kernel_NumWorkgroups_x>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimize[0] = __spirv_NumWorkgroups_x();
    });
    // CHECK-LABEL: @{{.*}}kernel_NumWorkgroups_x
    // CHECK: [[CALL_ESIMD:%.*]] = call <3 x i32> @llvm.genx.group.count.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD]], i32 0

    kernel<class kernel_NumWorkgroups_y>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimize[0] = __spirv_NumWorkgroups_y();
    });
    // CHECK-LABEL: @{{.*}}kernel_NumWorkgroups_y
    // CHECK: [[CALL_ESIMD:%.*]] = call <3 x i32> @llvm.genx.group.count.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD]], i32 1

    kernel<class kernel_NumWorkgroups_z>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimize[0] = __spirv_NumWorkgroups_z();
    });
    // CHECK-LABEL: @{{.*}}kernel_NumWorkgroups_z
    // CHECK: [[CALL_ESIMD:%.*]] = call <3 x i32> @llvm.genx.group.count.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD]], i32 2

    kernel<class kernel_WorkgroupSize_x>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimize[0] = __spirv_WorkgroupSize_x();
    });
    // CHECK-LABEL: @{{.*}}kernel_WorkgroupSize_x
    // CHECK: [[CALL_ESIMD:%.*]] = call <3 x i32> @llvm.genx.local.size.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD]], i32 0

    kernel<class kernel_WorkgroupSize_y>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimize[0] = __spirv_WorkgroupSize_y();
    });
    // CHECK-LABEL: @{{.*}}kernel_WorkgroupSize_y
    // CHECK: [[CALL_ESIMD:%.*]] = call <3 x i32> @llvm.genx.local.size.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD]], i32 1

    kernel<class kernel_WorkgroupSize_z>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimize[0] = __spirv_WorkgroupSize_z();
    });
    // CHECK-LABEL: @{{.*}}kernel_WorkgroupSize_z
    // CHECK: [[CALL_ESIMD:%.*]] = call <3 x i32> @llvm.genx.local.size.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD]], i32 2

    kernel<class kernel_WorkgroupId_x>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimize[0] = __spirv_WorkgroupId_x();
    });
    // CHECK-LABEL: @{{.*}}kernel_WorkgroupId_x
    // CHECK: {{.*}} call i32 @llvm.genx.group.id.x()

    kernel<class kernel_WorkgroupId_y>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimize[0] = __spirv_WorkgroupId_y();
    });
    // CHECK-LABEL: @{{.*}}kernel_WorkgroupId_y
    // CHECK: {{.*}} call i32 @llvm.genx.group.id.y()

    kernel<class kernel_WorkgroupId_z>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimize[0] = __spirv_WorkgroupId_z();
    });
    // CHECK-LABEL: @{{.*}}kernel_WorkgroupId_z
    // CHECK: {{.*}} call i32 @llvm.genx.group.id.z()

    kernel<class kernel_LocalInvocationId_x>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimize[0] = __spirv_LocalInvocationId_x();
    });
    // CHECK-LABEL: @{{.*}}kernel_LocalInvocationId_x
    // CHECK: [[CALL_ESIMD:%.*]] = call <3 x i32> @llvm.genx.local.id.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD]], i32 0

    kernel<class kernel_LocalInvocationId_y>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimize[0] = __spirv_LocalInvocationId_y();
    });
    // CHECK-LABEL: @{{.*}}kernel_LocalInvocationId_y
    // CHECK: [[CALL_ESIMD:%.*]] = call <3 x i32> @llvm.genx.local.id.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD]], i32 1

    kernel<class kernel_LocalInvocationId_z>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimize[0] = __spirv_LocalInvocationId_z();
    });
    // CHECK-LABEL: @{{.*}}kernel_LocalInvocationId_z
    // CHECK: [[CALL_ESIMD:%.*]] = call <3 x i32> @llvm.genx.local.id.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD]], i32 2

    // Tests below check correct translation of loads from SPIRV builtin
    // globals, when load has multiple uses, e.g.:
    //  %0 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64>
    //  addrspace(1)* @__spirv_BuiltInWorkgroupId to <3 x i64> addrspace(4)*),
    //  align 32 %1 = extractelement <3 x i64> %0, i64 0 %2 = extractelement <3
    //  x i64> %0, i64 1 %3 = extractelement <3 x i64> %0, i64 2
    // In this case we will generate 3 calls to the same GenX intrinsic,
    // But -early-cse will later remove this redundancy.
    auto DoNotOptimizeXYZ = &DoNotOptXYZ[0];
    kernel<class kernel_LocalInvocationId_xyz>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimizeXYZ[0] = __spirv_LocalInvocationId_x();
      DoNotOptimizeXYZ[1] = __spirv_LocalInvocationId_y();
      DoNotOptimizeXYZ[2] = __spirv_LocalInvocationId_z();
    });
    // CHECK-LABEL: @{{.*}}kernel_LocalInvocationId_xyz
    // CHECK: [[CALL_ESIMD1:%.*]] = call <3 x i32> @llvm.genx.local.id.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD1]], i32 0
    // CHECK: [[CALL_ESIMD2:%.*]] = call <3 x i32> @llvm.genx.local.id.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD2]], i32 1
    // CHECK: [[CALL_ESIMD3:%.*]] = call <3 x i32> @llvm.genx.local.id.v3i32()
    // CHECK: {{.*}} extractelement <3 x i32> [[CALL_ESIMD3]], i32 2

    kernel<class kernel_WorkgroupId_xyz>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimizeXYZ[0] = __spirv_WorkgroupId_x();
      DoNotOptimizeXYZ[1] = __spirv_WorkgroupId_y();
      DoNotOptimizeXYZ[2] = __spirv_WorkgroupId_z();
    });
    // CHECK-LABEL: @{{.*}}kernel_WorkgroupId_xyz
    // CHECK: {{.*}} call i32 @llvm.genx.group.id.x()
    // CHECK: {{.*}} call i32 @llvm.genx.group.id.y()
    // CHECK: {{.*}} call i32 @llvm.genx.group.id.z()

    kernel<class kernel_SubgroupLocalInvocationId>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimize[0] = __spirv_SubgroupLocalInvocationId();
      DoNotOptimize32[0] = __spirv_SubgroupLocalInvocationId() + 3;
    });
    // CHECK-LABEL: @{{.*}}kernel_SubgroupLocalInvocationId
    // CHECK: [[ZEXT0:%.*]] = zext i32 0 to i64
    // CHECK: store i64 [[ZEXT0]]
    // CHECK: add i32 0, 3

    kernel<class kernel_SubgroupSize>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimize[0] = __spirv_SubgroupSize();
      DoNotOptimize32[0] = __spirv_SubgroupSize() + 7;
    });
    // CHECK-LABEL: @{{.*}}kernel_SubgroupSize
    // CHECK: [[ZEXT0:%.*]] = zext i32 1 to i64
    // CHECK: store i64 [[ZEXT0]]
    // CHECK: add i32 1, 7

    kernel<class kernel_SubgroupMaxSize>([=]() SYCL_ESIMD_KERNEL {
      DoNotOptimize[0] = __spirv_SubgroupMaxSize();
      DoNotOptimize32[0] = __spirv_SubgroupMaxSize() + 9;
    });
    // CHECK-LABEL: @{{.*}}kernel_SubgroupMaxSize
    // CHECK: [[ZEXT0:%.*]] = zext i32 1 to i64
    // CHECK: store i64 [[ZEXT0]]
    // CHECK: add i32 1, 9
  });
  return DoNotOpt[0];
}
