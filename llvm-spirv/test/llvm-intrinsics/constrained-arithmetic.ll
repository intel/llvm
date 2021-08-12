; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv


; CHECK: Name [[ad:[0-9]+]] "add"
; CHECK: Name [[di:[0-9]+]] "div"
; CHECK: Name [[su:[0-9]+]] "sub"
; CHECK: Name [[mu:[0-9]+]] "mul"

; CHECK-NOT: Decorate {{[0-9]+}} FPRoundingMode

; CHECK: Decorate [[ad]] FPRoundingMode 0
; CHECK: Decorate [[di]] FPRoundingMode 1
; CHECK: Decorate [[su]] FPRoundingMode 2
; CHECK: Decorate [[mu]] FPRoundingMode 3

; CHECK-NOT: Decorate {{[0-9]+}} FPRoundingMode

; CHECK: FAdd {{[0-9]+}} [[ad]]
; CHECK: FDiv {{[0-9]+}} [[di]]
; CHECK: FSub {{[0-9]+}} [[su]]
; CHECK: FMul {{[0-9]+}} [[mu]]
; CHECK: FMul
; CHECK: FAdd
; CHECK: ExtInst {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} fma
; CHECK: FRem

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-linux-sycldevice"

; Function Attrs: norecurse nounwind strictfp
define dso_local spir_kernel void @test(float %a, i32 %in, i32 %ui) local_unnamed_addr #0 !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !7 !kernel_arg_base_type !7 !kernel_arg_type_qual !8 !kernel_arg_buffer_location !9 {
entry:
  %add = tail call float @llvm.experimental.constrained.fadd.f32(float %a, float %a, metadata !"round.tonearest", metadata !"fpexcept.strict") #2
  %div = tail call float @llvm.experimental.constrained.fdiv.f32(float %add, float %add, metadata !"round.towardzero", metadata !"fpexcept.strict") #2, !fpmath !10
  %sub = tail call float @llvm.experimental.constrained.fsub.f32(float %div, float %div, metadata !"round.upward", metadata !"fpexcept.strict") #2
  %mul = tail call float @llvm.experimental.constrained.fmul.f32(float %sub, float %sub, metadata !"round.downward", metadata !"fpexcept.strict") #2
  %0 = tail call float @llvm.experimental.constrained.fmuladd.f32(float %mul, float %mul, float %mul, metadata !"round.tonearestaway", metadata !"fpexcept.strict") #2
  %1 = tail call float @llvm.experimental.constrained.fma.f32(float %0, float %0, float %0, metadata !"round.dynamic", metadata !"fpexcept.strict") #2
  %2 = tail call float @llvm.experimental.constrained.frem.f32(float %1, float %1, metadata !"round.dynamic", metadata !"fpexcept.strict") #2
  ret void
}

; Function Attrs: inaccessiblememonly nounwind willreturn
declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata) #1

; Function Attrs: inaccessiblememonly nounwind willreturn
declare float @llvm.experimental.constrained.fdiv.f32(float, float, metadata, metadata) #1

; Function Attrs: inaccessiblememonly nounwind willreturn
declare float @llvm.experimental.constrained.fsub.f32(float, float, metadata, metadata) #1

; Function Attrs: inaccessiblememonly nounwind willreturn
declare float @llvm.experimental.constrained.fmul.f32(float, float, metadata, metadata) #1

; Function Attrs: inaccessiblememonly nounwind willreturn
declare float @llvm.experimental.constrained.fmuladd.f32(float, float, float, metadata, metadata) #1

; Function Attrs: inaccessiblememonly nounwind willreturn
declare float @llvm.experimental.constrained.fma.f32(float, float, float, metadata, metadata) #1

; Function Attrs: inaccessiblememonly nounwind willreturn
declare float @llvm.experimental.constrained.frem.f32(float, float, metadata, metadata) #1

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
!5 = !{i32 0, i32 0, i32 0}
!6 = !{!"none", !"none", !"none"}
!7 = !{!"float", !"int", !"uint"}
!8 = !{!"", !"", !""}
!9 = !{i32 -1, i32 -1, i32 -1}
!10 = !{float 2.500000e+00}
