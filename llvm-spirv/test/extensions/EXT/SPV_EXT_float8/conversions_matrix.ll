; This tests checks if FP8 matrix conversions work fine.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_EXT_float8,+SPV_KHR_cooperative_matrix
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -o %t.rev.bc -r --spirv-target-env=SPV-IR
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; TODO: RUNx: spirv-val

; CHECK-SPIRV-DAG: CooperativeMatrixKHR
; CHECK-SPIRV-DAG: Float8EXT
; CHECK-SPIRV-DAG: Float8CooperativeMatrixEXT
; CHECK-SPIRV-DAG: "SPV_EXT_float8"
; CHECK-SPIRV-DAG: "SPV_KHR_cooperative_matrix"

; CHECK-SPIRV-DAG: TypeInt [[#Int8Ty:]] 8 0
; CHECK-SPIRV-DAG: TypeFloat [[#FP8Ty:]] 8 4214
; CHECK-SPIRV-DAG: TypeFloat [[#FP16Ty:]] 16
; CHECK-SPIRV-DAG: TypeCooperativeMatrixKHR [[#Int8MatrixTy:]] [[#Int8Ty]]
; CHECK-SPIRV-DAG: TypeCooperativeMatrixKHR [[#FP8MatrixTy:]] [[#FP8Ty]]
; CHECK-SPIRV-DAG: TypeCooperativeMatrixKHR [[#FP16MatrixTy:]] [[#FP16Ty]]

; CHECK-SPIRV: CompositeConstruct [[#FP16MatrixTy]] [[#M:]] [[#]]
; CHECK-SPIRV: FConvert [[#FP8MatrixTy]] [[#Conv:]] [[#M]]
; CHECK-SPIRV: Bitcast [[#Int8MatrixTy]] [[#]] [[#Conv]]

; CHECK-LLVM: %[[#M:]] = call spir_func target("spirv.CooperativeMatrixKHR", half, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructDh(half 0xH0000)
; CHECK-LLVM: call target("spirv.CooperativeMatrixKHR", i8, 3, 12, 12, 2) @_Z36__builtin_spirv_ConvertFP16ToE4M3EXTPU3AS144__spirv_CooperativeMatrixKHR__half_3_12_12_2(target("spirv.CooperativeMatrixKHR", half, 3, 12, 12, 2) %[[#M]])

; ModuleID = 'test.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-G1"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind
define spir_func void @int4_hf8() #0 {
entry:
  %0 = call spir_func target("spirv.CooperativeMatrixKHR", half, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructDh(half 0.0) #0
  %1 = call target("spirv.CooperativeMatrixKHR", i8, 3, 12, 12, 2) @_Z36__builtin_spirv_ConvertFP16ToE4M3EXTPU3AS144__spirv_CooperativeMatrixKHR__half_3_12_12_2(target("spirv.CooperativeMatrixKHR", half, 3, 12, 12, 2) %0)
  ret void
}

; Function Attrs: nounwind
declare spir_func target("spirv.CooperativeMatrixKHR", half, 3, 12, 12, 2) @_Z26__spirv_CompositeConstructDh(half) #0

declare target("spirv.CooperativeMatrixKHR", i8, 3, 12, 12, 2) @_Z36__builtin_spirv_ConvertFP16ToE4M3EXTPU3AS144__spirv_CooperativeMatrixKHR__half_3_12_12_2(target("spirv.CooperativeMatrixKHR", half, 3, 12, 12, 2))

attributes #0 = { nounwind }

!spirv.MemoryModel = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!1}
!opencl.spir.version = !{!0}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!spirv.Generator = !{!3}

!0 = !{i32 1, i32 2}
!1 = !{i32 0, i32 0}
!2 = !{}
!3 = !{i16 6, i16 14}
