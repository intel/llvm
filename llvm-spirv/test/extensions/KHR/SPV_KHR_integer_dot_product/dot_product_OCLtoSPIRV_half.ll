; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv -s %t.bc -o %t.regularized.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_integer_dot_product -o %t-spirv.spv
; RUN: spirv-val %t-spirv.spv
; RUN: llvm-dis %t.regularized.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM
; RUN: llvm-spirv %t.bc -spirv-text --spirv-ext=+SPV_KHR_integer_dot_product -o - | FileCheck %s --check-prefix=CHECK-SPIRV

;CHECK-LLVM: fmul

;CHECK-SPIRV: FMul

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir"

; Function Attrs: convergent norecurse nounwind
define spir_kernel void @test1(half %ha, half %hb) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %call = tail call spir_func half @_Z3dotDhDh(half %ha, half %hb) #2
  ret void
}

; Function Attrs: convergent
declare spir_func half @_Z3dotDhDh(half, half) local_unnamed_addr #1

attributes #0 = { convergent norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pocharer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="128" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pocharer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 11.0.0 (https://github.com/c199914007/llvm.git 8b94769313ca84cb9370b60ed008501ff692cb71)"}
!3 = !{i32 0, i32 0}
!4 = !{!"none", !"none"}
!5 = !{!"half", !"half"}
!6 = !{!"half", !"half"}
!7 = !{!"", !""}
