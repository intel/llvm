; intel/llvm customization

; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_cooperative_matrix -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: TypeInt [[#CharTy:]] 8 0
; CHECK-SPIRV-DAG: TypePointer [[#CharPtrTy:]] 8 [[#CharTy]]
; CHECK-SPIRV-DAG: TypeCooperativeMatrixKHR [[#MatTy:]] [[#CharTy]]
; CHECK-SPIRV-DAG: TypePointer [[#PtrMatTy:]] 7 [[#MatTy]]
; CHECK-SPIRV: Variable [[#PtrMatTy]] [[#Var:]] 7
; CHECK-SPIRV: PtrCastToGeneric [[#]] [[#Cast:]] [[#Var]]
; CHECK-SPIRV: AccessChain [[#CharPtrTy]] [[#]] [[#Cast]] [[#]]

; CHECK-LLVM: %[[#Alloca:]] = alloca target("spirv.CooperativeMatrixKHR", i8, 3, 16, 64, 0)
; CHECK-LLVM: %[[#Cast:]] = addrspacecast ptr %[[#Alloca]] to ptr addrspace(4)
; CHECK-LLVM: call spir_func ptr addrspace(4) @_Z19__spirv_AccessChainPU3AS4PU3AS144__spirv_CooperativeMatrixKHR__char_3_16_64_0l(ptr addrspace(4) %[[#Cast]], i64 0)

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
