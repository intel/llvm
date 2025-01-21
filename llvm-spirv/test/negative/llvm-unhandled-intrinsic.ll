; Translator should not translate llvm intrinsic calls straight forward.
; It either represents intrinsic's semantics with SPIRV instruction(s), or
; reports an error.
; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv %t.bc 2>&1 | FileCheck %s

; CHECK: InvalidFunctionCall: Unexpected llvm intrinsic:
; CHECK-NEXT: llvm.readcyclecounter

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

; Function Attrs: nounwind
define spir_func void @foo() #0 {
entry:
  %0 = call i64 @llvm.readcyclecounter()
  ret void
}

declare i64 @llvm.readcyclecounter()

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!3}
!opencl.compiler.options = !{!2}

!0 = !{i32 1, i32 2}
!1 = !{i32 2, i32 0}
!2 = !{}
!3 = !{!"cl_doubles"}
