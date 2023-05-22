; RUN: mlir-translate -import-llvm %s | FileCheck %s

; CHECK: llvm.func spir_kernelcc @foo(%arg0: !llvm.target<"spirv.Pipe", 0>
; CHECK-SAME: %arg1: !llvm.target<"spirv.Pipe", 1>
; CHECK-SAME: %arg2: !llvm.target<"spirv.Image", !llvm.void, 0, 0, 0, 0, 0, 0, 0>
; CHECK-SAME: %arg3: !llvm.target<"spirv.Image", i32, 1, 0, 0, 0, 0, 0, 0>
; CHECK-SAME: %arg4: !llvm.target<"spirv.Image", i32, 2, 0, 0, 0, 0, 0, 0>
; CHECK-SAME: %arg5: !llvm.target<"spirv.Image", f16, 1, 0, 1, 0, 0, 0, 0>
; CHECK-SAME: %arg6: !llvm.target<"spirv.Image", f32, 5, 0, 0, 0, 0, 0, 0>
; CHECK-SAME: %arg7: !llvm.target<"spirv.Image", !llvm.void, 0, 0, 0, 0, 0, 0, 1>
; CHECK-SAME: %arg8: !llvm.target<"spirv.Image", !llvm.void, 1, 0, 0, 0, 0, 0, 2>)
define spir_kernel void @foo(
  target("spirv.Pipe", 0) %a,
  target("spirv.Pipe", 1) %b,
  target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 0) %c1,
  target("spirv.Image", i32, 1, 0, 0, 0, 0, 0, 0) %d1,
  target("spirv.Image", i32, 2, 0, 0, 0, 0, 0, 0) %e1,
  target("spirv.Image", half, 1, 0, 1, 0, 0, 0, 0) %f1,
  target("spirv.Image", float, 5, 0, 0, 0, 0, 0, 0) %g1,
  target("spirv.Image", void, 0, 0, 0, 0, 0, 0, 1) %c2,
  target("spirv.Image", void, 1, 0, 0, 0, 0, 0, 2) %d3) {
entry:
  ret void
}
