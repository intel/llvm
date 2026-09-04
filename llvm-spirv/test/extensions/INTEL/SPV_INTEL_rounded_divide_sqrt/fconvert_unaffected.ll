; Enabling SPV_INTEL_rounded_divide_sqrt shouldn't add RoundedDivideSqrtINTEL,
; as FPRoundingMode on conversion instructions is supported in core SPIR-V

; RUN: llvm-spirv %s -o %t.spv --spirv-ext=+SPV_INTEL_rounded_divide_sqrt
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck %s --input-file %t.spt

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK-NOT: Capability RoundedDivideSqrtINTEL
; CHECK-NOT: Extension "SPV_INTEL_rounded_divide_sqrt"
; The conversion still carries its rounding-mode decoration (core SPIR-V).
; CHECK: Decorate [[#CVT:]] FPRoundingMode 1
; CHECK: FConvert [[#]] [[#CVT]]

define spir_kernel void @test(double %a) {
entry:
  %r = call float @llvm.experimental.constrained.fptrunc.f32.f64(double %a, metadata !"round.towardzero", metadata !"fpexcept.strict")
  ret void
}

declare float @llvm.experimental.constrained.fptrunc.f32.f64(double, metadata, metadata)
