; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -spirv-text %t.bc -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: 3 Name [[#r1:]] "r1"
; CHECK-SPIRV: 3 Name [[#r2:]] "r2"
; CHECK-SPIRV: 3 Name [[#r3:]] "r3"
; CHECK-SPIRV: 3 Name [[#r4:]] "r4"
; CHECK-SPIRV: 3 Name [[#r5:]] "r5"
; CHECK-SPIRV: 3 Name [[#r6:]] "r6"
; CHECK-SPIRV: 3 Name [[#r7:]] "r7"
; CHECK-SPIRV-NOT: 4 Decorate {{.*}} FPFastMathMode
; CHECK-SPIRV: 3 TypeFloat [[float:[0-9]+]] 32
; CHECK-SPIRV: 4 FNegate [[float]] [[#r1]]
; CHECK-SPIRV: 4 FNegate [[float]] [[#r2]]
; CHECK-SPIRV: 4 FNegate [[float]] [[#r3]]
; CHECK-SPIRV: 4 FNegate [[float]] [[#r4]]
; CHECK-SPIRV: 4 FNegate [[float]] [[#r5]]
; CHECK-SPIRV: 4 FNegate [[float]] [[#r6]]
; CHECK-SPIRV: 4 FNegate [[float]] [[#r7]]

; CHECK-LLVM: %r1 = fneg float %a
; CHECK-LLVM: %r2 = fneg float %a
; CHECK-LLVM: %r3 = fneg float %a
; CHECK-LLVM: %r4 = fneg float %a
; CHECK-LLVM: %r5 = fneg float %a
; CHECK-LLVM: %r6 = fneg float %a
; CHECK-LLVM: %r7 = fneg float %a

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @testFNeg(float %a) local_unnamed_addr #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %r1 = fneg float %a
  %r2 = fneg nnan float %a
  %r3 = fneg ninf float %a
  %r4 = fneg nsz float %a
  %r5 = fneg arcp float %a
  %r6 = fneg fast float %a
  %r7 = fneg nnan ninf float %a
  ret void
}

attributes #0 = { convergent nounwind writeonly "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{i32 0}
!3 = !{!"none"}
!4 = !{!"float"}
!5 = !{!""}
