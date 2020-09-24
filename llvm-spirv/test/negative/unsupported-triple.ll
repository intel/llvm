; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv %t.bc -o %t.spv 2>&1 | FileCheck %s

; CHECK: InvalidTargetTriple: Expects spir-unknown-unknown or spir64-unknown-unknown. Actual target triple is aarch64

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "aarch64"

; Function Attrs: convergent noinline nounwind optnone
define spir_func void @_Z3foov() {
entry:
  ret void
}

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!opencl.ocl.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
