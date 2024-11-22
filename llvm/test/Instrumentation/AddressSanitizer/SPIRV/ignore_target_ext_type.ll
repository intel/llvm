; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 -asan-stack=0 -asan-globals=0 -asan-use-after-return=never -asan-stack-dynamic-alloca=0 -asan-mapping-scale=4 -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

%"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix" = type { target("spirv.JointMatrixINTEL", i16, 16, 32, 0, 3, 0, 1) }

define spir_kernel void @_ZTS4multIN4sycl3_V13ext6oneapi8bfloat16ELm16ELm16ELm32EE() sanitize_address {
entry:
; CHECK-LABEL: @_ZTS4multIN4sycl3_V13ext6oneapi8bfloat16ELm16ELm16ELm32EE
; CHECK-NOT: MyAlloc
  %a = alloca [2 x %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix"], i32 0, align 8
  br label %for.cond10.i

for.cond10.i:                                     ; preds = %for.cond10.i, %entry
  %0 = load target("spirv.JointMatrixINTEL", i16, 16, 32, 0, 3, 0, 1), ptr null, align 8
  store target("spirv.JointMatrixINTEL", float, 16, 16, 3, 3, 2) zeroinitializer, ptr null, align 8
; CHECK-NOT: call void @asan_load
; CHECK-NOT: call void @asan_store
  br label %for.cond10.i
}

define spir_kernel void @AccessChain() sanitize_address {
entry:
; CHECK-LABEL: @AccessChain
  %a = alloca %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix", align 8
  %0 = getelementptr inbounds %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix", ptr %a, i64 0, i32 0
  %call.i35 = call spir_func ptr @_Z19__spirv_AccessChainIfN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32ELm8ELm8ELN5__spv9MatrixUseE0ELNS8_5Scope4FlagE3EEPT_PPNS8_28__spirv_CooperativeMatrixKHRIT0_XT4_EXT1_EXT2_EXT3_EEEm(ptr %0, i64 0)
; CHECK-NOT: call void @__asan_load
; CHECK-NOT: call void @__asan_store
  %1 = load float, ptr %call.i35, align 4
  %call.i42 = call spir_func ptr @_Z19__spirv_AccessChainIfN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32ELm8ELm8ELN5__spv9MatrixUseE0ELNS8_5Scope4FlagE3EEPT_PPNS8_28__spirv_CooperativeMatrixKHRIT0_XT4_EXT1_EXT2_EXT3_EEEm(ptr %0, i64 0)
  store float %1, ptr %call.i42, align 4
  ret void
}

declare spir_func ptr @_Z19__spirv_AccessChainIfN4sycl3_V13ext6oneapi12experimental6matrix9precision4tf32ELm8ELm8ELN5__spv9MatrixUseE0ELNS8_5Scope4FlagE3EEPT_PPNS8_28__spirv_CooperativeMatrixKHRIT0_XT4_EXT1_EXT2_EXT3_EEEm(ptr, i64)
