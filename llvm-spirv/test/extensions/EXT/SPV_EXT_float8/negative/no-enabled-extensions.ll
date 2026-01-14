; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv %t.bc 2>&1 \
; RUN: | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-ERROR-NEXT: One of the following extensions: SPV_EXT_float8,
; CHECK-ERROR-SAME: SPV_INTEL_int4 should be enabled to process conversion builtins
; CHECK-ERROR-NEXT: declare dso_local spir_func i8 @_Z36__builtin_spirv_ConvertFP16ToE4M3EXTDh(half)

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

define spir_func i8 @fp16_hf8() {
entry:
  %0 = call spir_func i8 @_Z36__builtin_spirv_ConvertFP16ToE4M3EXTDh(half 0.0)
  ret i8 %0
}

declare dso_local spir_func i8 @_Z36__builtin_spirv_ConvertFP16ToE4M3EXTDh(half)
