; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_KHR_bfloat16 -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_KHR_bfloat16 -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: Capability CooperativeMatrixKHR
; CHECK-SPIRV-DAG: Capability BFloat16TypeKHR
; CHECK-SPIRV-DAG: Capability BFloat16CooperativeMatrixKHR
; CHECK-SPIRV-DAG: Extension "SPV_KHR_cooperative_matrix"
; CHECK-SPIRV-DAG: Extension "SPV_KHR_bfloat16"

; CHECK-SPIRV-DAG: 4 TypeFloat [[#BFloatTy:]] 16 0
; CHECK-SPIRV-DAG: TypeInt [[#Int32Ty:]] 32 0
; CHECK-SPIRV-DAG: Constant [[#Int32Ty]] [[#Const12:]] 12
; CHECK-SPIRV-DAG: Constant [[#Int32Ty]] [[#Const3:]] 3
; CHECK-SPIRV-DAG: Constant [[#Int32Ty]] [[#Const2:]] 2
; CHECK-SPIRV-DAG: TypeCooperativeMatrixKHR [[#MatTy:]] [[#BFloatTy]] [[#Const3]] [[#Const12]] [[#Const12]] [[#Const2]]
; CHECK-SPIRV-DAG: Constant [[#BFloatTy]] [[#]] 16256
; CHECK-SPIRV: CompositeConstruct [[#MatTy]]

; CHECK-LLVM: call spir_func target("spirv.CooperativeMatrixKHR", bfloat, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructu6__bf16(bfloat 0xR3F80)

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

declare spir_func target("spirv.CooperativeMatrixKHR", bfloat, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructu6__bf16(bfloat)

define spir_kernel void @test() {
  %mat = call spir_func target("spirv.CooperativeMatrixKHR", bfloat, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructu6__bf16(bfloat 1.0)
  ret void
}
