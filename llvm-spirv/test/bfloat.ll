; Check that translator emits error for LLVM bfloat type
; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv --spirv-ext=+all %t.bc -o %t.spv 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: UnsupportedLLVMBFloatType: LLVM bfloat type is not supported in SPIR-V

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @testBFloat(bfloat %a, bfloat %b) {
entry:
  %r1 = fmul bfloat %a, %b
  ret void
}

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
