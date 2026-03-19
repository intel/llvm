; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_INTEL_sigmoid
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -o %t.rev.bc -r --spirv-target-env=SPV-IR
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; RUN: not llvm-spirv %t.bc 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-ERROR-NEXT: SPV_INTEL_sigmoid

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV: Capability SigmoidINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_sigmoid"
; CHECK-SPIRV: TypeFloat [[#FP16Ty:]] 16
; CHECK-SPIRV: TypeVector [[#FP16v8Ty:]] [[#FP16Ty]] 8
; CHECK-SPIRV: Constant [[#FP16Ty]] [[#CONST:]] 15360

; CHECK-SPIRV: FunctionParameter [[#FP16Ty]] [[FP16ValId:.*]]
; CHECK-SPIRV: FunctionParameter [[#FP16v8Ty]] [[FP16v8ValId:.*]]

; CHECK-SPIRV: FSigmoidINTEL [[#FP16Ty]] [[#]] [[FP16ValId]]
; CHECK-SPIRV: FSigmoidINTEL [[#FP16v8Ty]] [[#]] [[FP16v8ValId]]
; CHECK-SPIRV: FSigmoidINTEL [[#FP16Ty]] [[#]] [[#CONST]]

; CHECK-LLVM: call spir_func half @_Z21__spirv_FSigmoidINTELDh(half
; CHECK-LLVM: call spir_func <8 x half> @_Z21__spirv_FSigmoidINTELDv8_Dh(<8 x half>
; CHECK-LLVM: call spir_func half @_Z21__spirv_FSigmoidINTELDh(half 0xH3C00)

define spir_func void @_Z2opffv8(half %a, <8 x half> %in) {
  %1 = tail call spir_func half @_Z21__spirv_FSigmoidINTELDh(half %a)
  %2 = tail call spir_func <8 x half> @_Z21__spirv_FSigmoidINTELDv8_Dh(<8 x half> %in)
  %3 = tail call spir_func half @_Z21__spirv_FSigmoidINTELDh(half 1.000000e+00)
  ret void
}

declare spir_func half @_Z21__spirv_FSigmoidINTELDh(half)

declare spir_func <8 x half> @_Z21__spirv_FSigmoidINTELDv8_Dh(<8 x half>)

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 16.0.0"}
