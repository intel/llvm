; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_INTEL_tensor_float32_conversion
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -o %t.rev.bc -r --spirv-target-env=SPV-IR
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; RUN: not llvm-spirv %t.bc 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-ERROR-NEXT: SPV_INTEL_tensor_float32_conversion

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV: Capability TensorFloat32RoundingINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_tensor_float32_conversion"
; CHECK-SPIRV: TypeFloat [[#FP32Ty:]] 32
; CHECK-SPIRV: TypeVector [[#FP32v8Ty:]] [[#FP32Ty]] 8
; CHECK-SPIRV: Constant [[#FP32Ty]] [[#CONST:]] 1065353216

; CHECK-SPIRV: FunctionParameter [[#FP32Ty]] [[FP32ValId:.*]]
; CHECK-SPIRV: FunctionParameter [[#FP32v8Ty]] [[FP32v8ValId:.*]]

; CHECK-SPIRV: RoundFToTF32INTEL [[#FP32Ty]] [[#]] [[FP32ValId]]
; CHECK-SPIRV: RoundFToTF32INTEL [[#FP32v8Ty]] [[#]] [[FP32v8ValId]]
; CHECK-SPIRV: RoundFToTF32INTEL [[#FP32Ty]] [[#]] [[#CONST]]

; CHECK-LLVM: call spir_func float @_Z25__spirv_RoundFToTF32INTELf(float
; CHECK-LLVM: call spir_func <8 x float> @_Z25__spirv_RoundFToTF32INTELDv8_f(<8 x float>
; CHECK-LLVM: call spir_func float @_Z25__spirv_RoundFToTF32INTELf(float 1.000000e+00)

define spir_func void @_Z2opffv8(float %a, <8 x float> %in) {
  %1 = tail call spir_func float @_Z25__spirv_RoundFToTF32INTELf(float %a)
  %2 = tail call spir_func <8 x float> @_Z25__spirv_RoundFToTF32INTELDv8_f(<8 x float> %in)
  %3 = tail call spir_func float @_Z25__spirv_RoundFToTF32INTELf(float 1.000000e+00)
  ret void
}

declare spir_func float @_Z25__spirv_RoundFToTF32INTELf(float)

declare spir_func <8 x float> @_Z25__spirv_RoundFToTF32INTELDv8_f(<8 x float>)

!opencl.spir.version = !{!0}
!spirv.Source = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, i32 2}
!1 = !{i32 4, i32 100000}
!2 = !{!"clang version 16.0.0"}
