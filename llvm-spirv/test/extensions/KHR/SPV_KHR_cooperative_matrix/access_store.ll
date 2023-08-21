; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_cooperative_matrix -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; TODO: come up with an approach and implement reverse translation
; R/UN: llvm-spirv -r %t.spv -o %t.rev.bc
; R/UN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: TypeInt [[#TypeInt:]] 32 0
; CHECK-SPIRV-DAG: Constant [[#TypeInt]] [[#Const0:]] 0
; CHECK-SPIRV-DAG: Constant [[#TypeInt]] [[#Const1:]] 1 {{$}}
; CHECK-SPIRV-DAG: Constant [[#TypeInt]] [[#Const3:]] 3
; CHECK-SPIRV-DAG: Constant [[#TypeInt]] [[#Const12:]] 12
; CHECK-SPIRV-DAG: Constant [[#TypeInt]] [[#Const42:]] 42

; CHECK-SPIRV: TypeCooperativeMatrixKHR [[#TypeMatrix:]] [[#TypeInt]] [[#Const3]] [[#Const12]] [[#Const12]] [[#Const0]]
; CHECK-SPIRV: TypePointer [[#Type:]] 7 [[#TypeInt]]

; CHECK-SPIRV: CompositeConstruct [[#TypeMatrix]] [[#Composite:]] [[#Const0]]
; CHECK-SPIRV: AccessChain [[#Type]] [[#Res:]] [[#Composite]] [[#Const1]]
; CHECK-SPIRV: Store [[#Res]] [[#Const42]]


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: mustprogress uwtable
define dso_local void @_Z3fooi(i32 noundef %idx) local_unnamed_addr #0 {
entry:
  %Obj = tail call spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 0) @_Z26__spirv_CompositeConstruct(i32 noundef 0) #4
  %call = call noundef ptr @_Z19__spirv_AccessChainP6Matrixii(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 0) %Obj, i32 noundef 1)
  call void @_Z13__spirv_StorePii(ptr noundef %call, i32 noundef 42)
  ret void
}

declare dso_local spir_func noundef target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 0) @_Z26__spirv_CompositeConstruct(i32 noundef) local_unnamed_addr #2

declare noundef ptr @_Z19__spirv_AccessChainP6Matrixii(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 0) noundef, i32 noundef) local_unnamed_addr #2

declare void @_Z13__spirv_StorePii(ptr noundef, i32 noundef) local_unnamed_addr #2

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
