; Before spirv-1.6 FNegate did not support FPFastMathMode flags

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -spirv-text %t.bc --spirv-max-version=1.5 -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-15
; RUN: llvm-spirv %t.bc --spirv-max-version=1.5 -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv -spirv-text %t.bc --spirv-max-version=1.6 -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-16,CHECK-SPIRV-16-DEFAULT
; RUN: llvm-spirv -spirv-text %t.bc --spirv-max-version=1.6 --spirv-ext=+SPV_KHR_float_controls2 -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-16,CHECK-SPIRV-16-FC2
; RUN: llvm-spirv %t.bc --spirv-max-version=1.6 -o %t.spv
; RUN: llvm-spirv %t.bc --spirv-max-version=1.6 --spirv-ext=+SPV_KHR_float_controls2 -o %t.fc2.spv
; RUN: spirv-val %t.spv
; RUN: spirv-val %t.fc2.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefixes=CHECK-LLVM-16,CHECK-LLVM-16-DEFAULT
; RUN: llvm-spirv -r %t.fc2.spv -o - | llvm-dis -o - | FileCheck %s --check-prefixes=CHECK-LLVM-16,CHECK-LLVM-16-FC2
; RUN: %if spirv-backend %{ llc -O0 -mtriple=spirv32-unknown-unknown -filetype=obj %s -o %t.llc.spv %}
; RUN: %if spirv-backend %{ llvm-spirv -r %t.llc.spv -o %t.llc.rev.bc %}
; RUN: %if spirv-backend %{ llvm-dis %t.llc.rev.bc -o %t.llc.rev.ll %}
; RUN: %if spirv-backend %{ FileCheck %s --check-prefix=CHECK-LLVM < %t.llc.rev.ll %}

; CHECK-SPIRV: 3 Name [[#r1:]] "r1"
; CHECK-SPIRV: 3 Name [[#r2:]] "r2"
; CHECK-SPIRV: 3 Name [[#r3:]] "r3"
; CHECK-SPIRV: 3 Name [[#r4:]] "r4"
; CHECK-SPIRV: 3 Name [[#r5:]] "r5"
; CHECK-SPIRV: 3 Name [[#r6:]] "r6"
; CHECK-SPIRV: 3 Name [[#r7:]] "r7"
; CHECK-SPIRV: 3 Name [[#r8:]] "r8"
; CHECK-SPIRV: 3 Name [[#r9:]] "r9"
; CHECK-SPIRV-15-NOT: 4 Decorate {{.*}} FPFastMathMode

; CHECK-SPIRV-16-NOT: Decorate [[#r1]] FPFastMathMode
; CHECK-SPIRV-16-DAG: Decorate [[#r2]] FPFastMathMode 1
; CHECK-SPIRV-16-DAG: Decorate [[#r3]] FPFastMathMode 2
; CHECK-SPIRV-16-DAG: Decorate [[#r4]] FPFastMathMode 4
; CHECK-SPIRV-16-DAG: Decorate [[#r5]] FPFastMathMode 8
; CHECK-SPIRV-16-DEFAULT-DAG: Decorate [[#r6]] FPFastMathMode 16
; CHECK-SPIRV-16-FC2-DAG: Decorate [[#r6]] FPFastMathMode 458767
; CHECK-SPIRV-16-DAG: Decorate [[#r7]] FPFastMathMode 3
; CHECK-SPIRV-16-DEFAULT-NOT: Decorate [[#r8]] FPFastMathMode
; CHECK-SPIRV-16-FC2-DAG: Decorate [[#r8]] FPFastMathMode 65536
; CHECK-SPIRV-16-DEFAULT-NOT: Decorate [[#r9]] FPFastMathMode
; CHECK-SPIRV-16-FC2-DAG: Decorate [[#r9]] FPFastMathMode 458752

; CHECK-SPIRV: 3 TypeFloat [[float:[0-9]+]] 32
; CHECK-SPIRV: 4 FNegate [[float]] [[#r1]]
; CHECK-SPIRV: 4 FNegate [[float]] [[#r2]]
; CHECK-SPIRV: 4 FNegate [[float]] [[#r3]]
; CHECK-SPIRV: 4 FNegate [[float]] [[#r4]]
; CHECK-SPIRV: 4 FNegate [[float]] [[#r5]]
; CHECK-SPIRV: 4 FNegate [[float]] [[#r6]]
; CHECK-SPIRV: 4 FNegate [[float]] [[#r7]]
; CHECK-SPIRV: 4 FNegate [[float]] [[#r8]]
; CHECK-SPIRV: 4 FNegate [[float]] [[#r9]]

; CHECK-LLVM: fneg float %a
; CHECK-LLVM: fneg float %a
; CHECK-LLVM: fneg float %a
; CHECK-LLVM: fneg float %a
; CHECK-LLVM: fneg float %a
; CHECK-LLVM: fneg float %a
; CHECK-LLVM: fneg float %a
; CHECK-LLVM: fneg float %a
; CHECK-LLVM: fneg float %a

; CHECK-LLVM-16: %r1 = fneg float %a
; CHECK-LLVM-16: %r2 = fneg nnan float %a
; CHECK-LLVM-16: %r3 = fneg ninf float %a
; CHECK-LLVM-16: %r4 = fneg nsz float %a
; CHECK-LLVM-16: %r5 = fneg arcp float %a
; CHECK-LLVM-16-DEFAULT: %r6 = fneg fast float %a
; CHECK-LLVM-16-FC2: %r6 = fneg reassoc nnan ninf nsz arcp contract float %a
; CHECK-LLVM-16: %r7 = fneg nnan ninf float %a
; CHECK-LLVM-16-DEFAULT: %r8 = fneg float %a
; CHECK-LLVM-16-FC2: %r8 = fneg contract float %a
; CHECK-LLVM-16-DEFAULT: %r9 = fneg float %a
; CHECK-LLVM-16-FC2: %r9 = fneg reassoc contract float %a

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @testFNeg(float %a) local_unnamed_addr #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %tmp = alloca float, align 4
  %r1 = fneg float %a
  store volatile float %r1, ptr %tmp, align 4
  %r2 = fneg nnan float %a
  store volatile float %r2, ptr %tmp, align 4
  %r3 = fneg ninf float %a
  store volatile float %r3, ptr %tmp, align 4
  %r4 = fneg nsz float %a
  store volatile float %r4, ptr %tmp, align 4
  %r5 = fneg arcp float %a
  store volatile float %r5, ptr %tmp, align 4
  %r6 = fneg fast float %a
  store volatile float %r6, ptr %tmp, align 4
  %r7 = fneg nnan ninf float %a
  store volatile float %r7, ptr %tmp, align 4
  %r8 = fneg contract float %a
  store volatile float %r8, ptr %tmp, align 4
  %r9 = fneg reassoc float %a
  store volatile float %r9, ptr %tmp, align 4
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
