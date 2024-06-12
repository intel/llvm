; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_KHR_shader_clock
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -o %t.rev.bc -r --spirv-target-env=SPV-IR
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; RUN: not llvm-spirv %t.bc 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-ERROR-NEXT: SPV_KHR_shader_clock

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV: Capability ShaderClockKHR
; CHECK-SPIRV: Extension "SPV_KHR_shader_clock"
; CHECK-SPIRV: TypeInt [[#I32Ty:]] 32
; CHECK-SPIRV: TypeInt [[#I64Ty:]] 64
; CHECK-SPIRV: TypeVector [[#I32v2Ty:]] [[#I32Ty]] 2

; CHECK-SPIRV: FunctionParameter [[#I32Ty]] [[I32ValId:.*]]

; CHECK-SPIRV: ReadClockKHR [[#I32v2Ty]] [[#]] [[I32ValId]]
; CHECK-SPIRV: ReadClockKHR [[#I64Ty]] [[#]] [[I32ValId]]

; CHECK-LLVM: call spir_func <2 x i32> @_Z27__spirv_ReadClockKHR_Ruint2i(
; CHECK-LLVM: call spir_func i64 @_Z27__spirv_ReadClockKHR_Rulongi(

define spir_func void @_Z7read_types(i32 %a) {
  %1 = tail call spir_func <2 x i32> @_Z20__spirv_ReadClockKHRIDv2_jET_j(i32 %a)
  %2 = tail call spir_func i64 @_Z20__spirv_ReadClockKHRImET_j(i32 %a)
  ret void
}

declare spir_func <2 x i32> @_Z20__spirv_ReadClockKHRIDv2_jET_j(i32)

declare spir_func i64 @_Z20__spirv_ReadClockKHRImET_j(i32)

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 16.0.0"}
