; RUN: opt < %s -passes=msan -msan-instrumentation-with-call-threshold=0 -msan-eager-checks=1 -msan-spir-privates=0 -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

%"class.sycl::_V1::ext::oneapi::bfloat16" = type { i16 }
%"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix" = type { target("spirv.CooperativeMatrixKHR", i16, 3, 16, 32, 0) }

; CHECK-LABEL: @test
; CHECK-NOT: call i64 @__msan_get_shadow
declare dso_local spir_func noundef ptr addrspace(4) @_Z19__spirv_AccessChainIN4sycl3_V13ext6oneapi8bfloat16ES4_Lm16ELm32ELN5__spv9MatrixUseE0ELNS5_5Scope4FlagE3EEPT_PPNS5_28__spirv_CooperativeMatrixKHRIT0_XT4_EXT1_EXT2_EXT3_EEEm(ptr addrspace(4) noundef, i64 noundef)

define weak_odr dso_local spir_kernel void @test() {
entry:
  %sub_a.i = alloca %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix", align 8
  %element.i = alloca %"class.sycl::_V1::ext::oneapi::bfloat16", align 2
  %0 = getelementptr inbounds { i16 }, ptr %element.i, i64 0, i32 0
  %spvm.i = getelementptr inbounds %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix", ptr %sub_a.i, i64 0, i32 0
  %addrcast = addrspacecast ptr %spvm.i to ptr addrspace(4)
  %call.i67 = call spir_func noundef ptr addrspace(4) @_Z19__spirv_AccessChainIN4sycl3_V13ext6oneapi8bfloat16ES4_Lm16ELm32ELN5__spv9MatrixUseE0ELNS5_5Scope4FlagE3EEPT_PPNS5_28__spirv_CooperativeMatrixKHRIT0_XT4_EXT1_EXT2_EXT3_EEEm(ptr addrspace(4) noundef %addrcast, i64 1)
  %gep = getelementptr inbounds nuw { i16 }, ptr addrspace(4) %call.i67, i64 0, i32 0
  %val = load i16, ptr %0, align 2
  store i16 %val, ptr addrspace(4) %gep, align 2
  ret void
}
