; RUN: llc -mtriple=spirv64-unknown-unknown -o - %s
; REQUIRES: spirv-registered-target
; Test that codegen doesn't crash for fpbuiltin intrinsic lowering.

define float @test() {
entry:
  %r = tail call float @llvm.fpbuiltin.fdiv.f32(float 1.000000e+00, float 2.000000e+00)
  ret float %r
}

define spir_kernel void @foo() {
entry:
  %r = tail call float @test()
  ret void
}
