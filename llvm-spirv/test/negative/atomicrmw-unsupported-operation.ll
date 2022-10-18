; RUN: llvm-as < %s -o %t.bc
; RUN: not llvm-spirv %t.bc -o %t.spv 2>&1 | FileCheck %s

; CHECK: InvalidInstruction: Can't translate llvm instruction:
; CHECK: Atomic nand is not supported in SPIR-V!
; CHECK: atomicrmw nand i32 addrspace(1)* @ui, i32 42 acq_rel

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

@ui = common dso_local addrspace(1) global i32 0, align 4
@f = common dso_local local_unnamed_addr addrspace(1) global float 0.000000e+00, align 4

; Function Attrs: nounwind
define dso_local spir_func void @test_atomicrmw() local_unnamed_addr #0 {
entry:
  %0 = atomicrmw nand i32 addrspace(1)* @ui, i32 42 acq_rel
  %1 = atomicrmw fadd float addrspace(1)* @f, float 42.000000e+00 seq_cst
  %2 = atomicrmw fsub float addrspace(1)* @f, float 42.000000e+00 seq_cst
  ret void
}

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 20c5968e0953d978be4d9d1062801e8758c393b5)"}
