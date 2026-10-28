; Check that an SPV-IR round trip does not duplicate SaturatedConversion.

; RUN: llvm-spirv %s -o %t.spv
; RUN: llvm-spirv -r --spirv-target-env=SPV-IR %t.spv -o %t.rev.bc
; RUN: llvm-spirv %t.rev.bc -o %t.rt.spv
; RUN: llvm-spirv --to-text %t.rt.spv -o - | FileCheck %s --implicit-check-not=SaturatedConversion
; RUN: spirv-val %t.rt.spv

; CHECK: Decorate [[#]] SaturatedConversion

target triple = "spir64-unknown-unknown"

define spir_func i32 @f(float %x) {
  %r = call i32 @llvm.fptosi.sat.i32.f32(float %x)
  ret i32 %r
}

declare i32 @llvm.fptosi.sat.i32.f32(float)
