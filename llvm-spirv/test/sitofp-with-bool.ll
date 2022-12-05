; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

; CHECK: TypeInt [[int_32:[0-9]+]] 32 0
; CHECK: Constant  {{[0-9]+}} [[zero:[0-9]+]] 0
; CHECK: Constant  {{[0-9]+}} [[one:[0-9]+]] 1
; CHECK: TypeBool [[bool:[0-9]+]]

; CHECK: Function
; CHECK: FunctionParameter {{[0-9]+}} [[A:[0-9]+]]
; CHECK: FunctionParameter {{[0-9]+}} [[B:[0-9]+]]
; CHECK: SGreaterThan [[bool]] [[cmp_res:[0-9]+]] [[B]] [[zero]]
; CHECK: Select [[int_32]] [[select_res:[0-9]+]] [[cmp_res]] [[one]] [[zero]]
; CHECK: ConvertSToF {{[0-9]+}} [[stof_res:[0-9]+]] [[select_res]]
; CHECK: Store [[A]] [[stof_res]]


target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

; Function Attrs: nofree norecurse nounwind writeonly
define dso_local spir_kernel void @K(float addrspace(1)* nocapture %A, i32 %B) local_unnamed_addr #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %cmp = icmp sgt i32 %B, 0
  %conv = sitofp i1 %cmp to float
  store float %conv, float addrspace(1)* %A, align 4
  ret void
}

attributes #0 = { nofree norecurse nounwind writeonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{i32 1, i32 0}
!3 = !{!"none", !"none"}
!4 = !{!"float*", !"int"}
!5 = !{!"", !""}
