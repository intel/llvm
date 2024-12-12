; The test checks, that unused call to __spirv_AccessChain is eliminated.

; RUN: opt -passes=sycl-joint-matrix-transform < %s -S | FileCheck %s

; CHECK-NOT: call spir_func ptr addrspace(4) @_Z19__spirv_AccessChain

; ModuleID = 'test.bc'
source_filename = "test.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

%"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix" = type { target("spirv.CooperativeMatrixKHR", i8, 3, 16, 64, 0) }

define weak_odr dso_local spir_kernel void @test() {
entry:
  %0 = alloca %"struct.sycl::_V1::ext::oneapi::experimental::matrix::joint_matrix", align 8
  %1 = addrspacecast ptr %0 to ptr addrspace(4)
  %2 = call spir_func ptr addrspace(4) @_Z19__spirv_AccessChainIiiLm16ELm16ELN5__spv9MatrixUseE2ELNS0_5Scope4FlagE3EEPT_PPNS0_28__spirv_CooperativeMatrixKHRIT0_XT4_EXT1_EXT2_EXT3_EEEm(ptr addrspace(4) noundef %1, i64 noundef 0)
  ret void
}

declare dso_local spir_func ptr addrspace(4) @_Z19__spirv_AccessChainIiiLm16ELm16ELN5__spv9MatrixUseE2ELNS0_5Scope4FlagE3EEPT_PPNS0_28__spirv_CooperativeMatrixKHRIT0_XT4_EXT1_EXT2_EXT3_EEEm(ptr addrspace(4) noundef, i64 noundef)
