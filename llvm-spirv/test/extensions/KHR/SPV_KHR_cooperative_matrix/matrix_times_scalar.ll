; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_cooperative_matrix -o %t.spv
; TODO: Validation is disabled till the moment the tools in CI are updated
; R/UN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: TypeFloat [[#TypeFloat:]] 32
; CHECK-SPIRV: TypeCooperativeMatrixKHR [[#MatrixType:]]

; CHECK-SPIRV: CompositeConstruct [[#MatrixType]] [[#Matrix:]] [[#]] {{$}}
; CHECK-SPIRV: Load [[#TypeFloat]] [[#Scalar:]]
; CHECK-SPIRV: MatrixTimesScalar [[#MatrixType]] [[#]] [[#Matrix]] [[#Scalar]]

; CHECK-LLVM: %[[#Matrix:]] = call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 3) @_Z26__spirv_CompositeConstructf(float 0.000000e+00)
; CHECK-LLVM: %[[#Scalar:]] = load float, ptr %scalar
; CHECK-LLVM: call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 3) @_Z25__spirv_MatrixTimesScalarPU3AS145__spirv_CooperativeMatrixKHR__float_3_12_12_3f(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 3) %[[#Matrix]], float %[[#Scalar]])

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: mustprogress uwtable
define dso_local void @matrix_times_scalar(ptr %scalar) local_unnamed_addr #0 {
entry:
  %0 = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 3) @_Z26__spirv_CompositeConstruct(float 0.000000e+00) #4
  %1 = load float, ptr %scalar, align 4
  %call = call noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 3) @_Z25__spirv_MatrixTimesScalar(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 3) %0, float %1)
  ret void
}

declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 3) @_Z26__spirv_CompositeConstruct(float noundef) local_unnamed_addr #2

declare noundef target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 3) @_Z25__spirv_MatrixTimesScalar(target("spirv.CooperativeMatrixKHR", float, 3, 12, 12, 3) noundef, float noundef) local_unnamed_addr #2

attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 7, !"Dwarf Version", i32 4}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"PIE Level", i32 2}
!4 = !{i32 7, !"uwtable", i32 2}
!5 = !{!"clang version 16.0.0 (https://github.com/llvm/llvm-project.git 08d094a0e457360ad8b94b017d2dc277e697ca76)"}
