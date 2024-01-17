; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_INTEL_joint_matrix,+SPV_INTEL_tensor_float32_conversion -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: not llvm-spirv %t.bc --spirv-ext=+SPV_KHR_cooperative_matrix,+SPV_INTEL_tensor_float32_conversion 2>&1 \
; RUN: | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: InvalidInstruction: Can't translate llvm instruction:
; CHECK-ERROR-NEXT: RoundFToTF32INTEL
; CHECK-ERROR-NEXT: Can be used with cooperative matrices only when SPV_INTEL_joint_matrix is enabled

; CHECK-SPIRV-DAG: Capability CooperativeMatrixKHR
; CHECK-SPIRV-DAG: Capability TensorFloat32RoundingINTEL
; CHECK-SPIRV-DAG: Capability JointMatrixTF32ComponentTypeINTEL
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_tensor_float32_conversion"
; CHECK-SPIRV-DAG: Extension "SPV_KHR_cooperative_matrix"
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_joint_matrix"
; CHECK-SPIRV-DAG: TypeFloat [[#FP32Ty:]] 32
; CHECK-SPIRV-DAG: TypeCooperativeMatrixKHR [[#FP32MatTy:]] [[#FP32Ty]]
; CHECK-SPIRV: CompositeConstruct [[#FP32MatTy]] [[#FP32Mat:]]
; CHECK-SPIRV: RoundFToTF32INTEL [[#FP32MatTy]] [[#]] [[#FP32Mat]]

; CHECK-LLVM: %[[#Mat:]] = call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructf(float 0.000000e+00)
; CHECK-LLVM: call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z25__spirv_RoundFToTF32INTELPU3AS145__spirv_CooperativeMatrixKHR__float_3_12_12_2(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) %[[#Mat]])


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define void @convert_f_to_tf() {
entry:
  %0 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructFloat(float 0.000000e+00)
  %call = call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z25__spirv_RoundFToTF32INTEL(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) %0)
  ret void
}

declare spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructFloat(float noundef)

declare spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) @_Z25__spirv_RoundFToTF32INTEL(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 2) noundef)

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 7, !"Dwarf Version", i32 4}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"PIE Level", i32 2}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{!"clang version 17.0.0"}
