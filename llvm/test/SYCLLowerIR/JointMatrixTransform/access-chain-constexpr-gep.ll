; Test checks that a constant-expression GEP on the __spirv_AccessChain operand
; is looked through without crashing, and that the IR is left unchanged since
; the base pointer is a global rather than an alloca.

; RUN: opt -passes=sycl-joint-matrix-transform < %s -S | FileCheck %s

; CHECK: %[[#AC:]] = call spir_func ptr addrspace(4) @_Z19__spirv_AccessChain{{.*}}(ptr addrspace(4) noundef addrspacecast (ptr addrspace(1) getelementptr inbounds ([4 x %"struct.sycl::_V1::long"], ptr addrspace(1) @arr, i64 0, i64 2) to ptr addrspace(4)), i64 noundef 0)
; CHECK: load i64, ptr addrspace(4) %[[#AC]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

%"struct.sycl::_V1::long" = type { i64 }

@arr = internal addrspace(1) global [4 x %"struct.sycl::_V1::long"] zeroinitializer

define weak_odr dso_local spir_kernel void @test() {
entry:
  %0 = call spir_func ptr addrspace(4) @_Z19__spirv_AccessChainIiiLm16ELm16ELN5__spv9MatrixUseE2ELNS0_5Scope4FlagE3EEPT_PPNS0_28__spirv_CooperativeMatrixKHRIT0_XT4_EXT1_EXT2_EXT3_EEEm(ptr addrspace(4) noundef addrspacecast (ptr addrspace(1) getelementptr inbounds ([4 x %"struct.sycl::_V1::long"], ptr addrspace(1) @arr, i64 0, i64 2) to ptr addrspace(4)), i64 noundef 0)
  %1 = load i64, ptr addrspace(4) %0
  ret void
}

declare dso_local spir_func ptr addrspace(4) @_Z19__spirv_AccessChainIiiLm16ELm16ELN5__spv9MatrixUseE2ELNS0_5Scope4FlagE3EEPT_PPNS0_28__spirv_CooperativeMatrixKHRIT0_XT4_EXT1_EXT2_EXT3_EEEm(ptr addrspace(4) noundef, i64 noundef)
