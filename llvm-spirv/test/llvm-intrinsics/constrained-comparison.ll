; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

;CHECK: FOrdEqual
;CHECK: FOrdGreaterThan
;CHECK: FOrdGreaterThanEqual
;CHECK: FOrdLessThan
;CHECK: FOrdLessThanEqual
;CHECK: FOrdNotEqual
;CHECK: Ordered
;CHECK: FUnordEqual
;CHECK: FUnordGreaterThan
;CHECK: FUnordGreaterThanEqual
;CHECK: FUnordLessThan
;CHECK: FUnordLessThanEqual
;CHECK: FUnordNotEqual
;CHECK: Unordered

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-linux"

; Function Attrs: norecurse nounwind strictfp
define dso_local spir_kernel void @test(float %a) local_unnamed_addr #0 !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !7 !kernel_arg_base_type !7 !kernel_arg_type_qual !8 !kernel_arg_buffer_location !9 {
entry:
  %cmp = tail call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %a, metadata !"oeq", metadata !"fpexcept.strict") #2
  %cmp1 = tail call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %a, metadata !"ogt", metadata !"fpexcept.strict") #2
  %cmp2 = tail call i1 @llvm.experimental.constrained.fcmps.f32(float %a, float %a, metadata !"oge", metadata !"fpexcept.strict") #2
  %cmp3 = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %a, metadata !"olt", metadata !"fpexcept.strict") #2
  %cmp4 = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %a, metadata !"ole", metadata !"fpexcept.strict") #2
  %cmp5 = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %a, metadata !"one", metadata !"fpexcept.strict") #2
  %cmp6 = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %a, metadata !"ord", metadata !"fpexcept.strict") #2
  %cmp7 = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %a, metadata !"ueq", metadata !"fpexcept.strict") #2
  %cmp8 = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %a, metadata !"ugt", metadata !"fpexcept.strict") #2
  %cmp9 = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %a, metadata !"uge", metadata !"fpexcept.strict") #2
  %cmp10 = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %a, metadata !"ult", metadata !"fpexcept.strict") #2
  %cmp11 = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %a, metadata !"ule", metadata !"fpexcept.strict") #2
  %cmp12 = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %a, metadata !"une", metadata !"fpexcept.strict") #2
  %cmp13 = tail call i1 @llvm.experimental.constrained.fcmp.f32(float %a, float %a, metadata !"uno", metadata !"fpexcept.strict") #2
  ret void
}

; Function Attrs: inaccessiblememonly nounwind willreturn
declare i1 @llvm.experimental.constrained.fcmps.f32(float, float, metadata, metadata) #1

; Function Attrs: inaccessiblememonly nounwind willreturn
declare i1 @llvm.experimental.constrained.fcmp.f32(float, float, metadata, metadata) #1

attributes #0 = { norecurse nounwind strictfp "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="test2.cl" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { inaccessiblememonly nounwind willreturn }
attributes #2 = { strictfp }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!2, !2}
!spirv.Source = !{!3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"clang version 12.0.0 (https://github.com/c199914007/llvm.git f0c85a8adeb49638c01eee1451aa9b35462cbfd5)"}
!5 = !{i32 0}
!6 = !{!"none"}
!7 = !{!"float"}
!8 = !{!""}
!9 = !{i32 -1}
