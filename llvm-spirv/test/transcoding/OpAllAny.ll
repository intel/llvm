; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.spv.bc --spirv-target-env=SPV-IR
; RUN: llvm-dis < %t.rev.spv.bc | FileCheck %s --check-prefix=CHECK-SPV-IR
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; This test checks SYCL relational builtin any and all with vector input types.

; CHECK-SPIRV: 2 TypeBool [[BoolTypeID:[0-9]+]]

; CHECK-SPIRV: 4 Any [[BoolTypeID]]
; CHECK-SPIRV: 4 Any [[BoolTypeID]]
; CHECK-SPIRV: 4 Any [[BoolTypeID]]
; CHECK-SPIRV: 4 Any [[BoolTypeID]]
; CHECK-SPIRV: 4 All [[BoolTypeID]]
; CHECK-SPIRV: 4 All [[BoolTypeID]]
; CHECK-SPIRV: 4 All [[BoolTypeID]]
; CHECK-SPIRV: 4 All [[BoolTypeID]]

; CHECK-SPV-IR: call spir_func i1 @_Z11__spirv_AnyDv2_c(<2 x i8>
; CHECK-SPV-IR: call spir_func i1 @_Z11__spirv_AnyDv2_c(<2 x i8>
; CHECK-SPV-IR: call spir_func i1 @_Z11__spirv_AnyDv2_c(<2 x i8>
; CHECK-SPV-IR: call spir_func i1 @_Z11__spirv_AnyDv2_c(<2 x i8>
; CHECK-SPV-IR: call spir_func i1 @_Z11__spirv_AllDv2_c(<2 x i8>
; CHECK-SPV-IR: call spir_func i1 @_Z11__spirv_AllDv2_c(<2 x i8>
; CHECK-SPV-IR: call spir_func i1 @_Z11__spirv_AllDv2_c(<2 x i8>
; CHECK-SPV-IR: call spir_func i1 @_Z11__spirv_AllDv2_c(<2 x i8>

; CHECK-LLVM: call spir_func i32 @_Z3anyDv2_c(<2 x i8>
; CHECK-LLVM: call spir_func i32 @_Z3anyDv2_c(<2 x i8>
; CHECK-LLVM: call spir_func i32 @_Z3anyDv2_c(<2 x i8>
; CHECK-LLVM: call spir_func i32 @_Z3anyDv2_c(<2 x i8>
; CHECK-LLVM: call spir_func i32 @_Z3allDv2_c(<2 x i8>
; CHECK-LLVM: call spir_func i32 @_Z3allDv2_c(<2 x i8>
; CHECK-LLVM: call spir_func i32 @_Z3allDv2_c(<2 x i8>
; CHECK-LLVM: call spir_func i32 @_Z3allDv2_c(<2 x i8>

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn writeonly
define dso_local spir_func void @test_vector(i32 addrspace(4)* nocapture writeonly %out, <2 x i8> %c, <2 x i16> %s, <2 x i32> %i, <2 x i64> %l) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func i32 @_Z3anyDv2_c(<2 x i8> %c) #2
  %call1 = tail call spir_func i32 @_Z3anyDv2_s(<2 x i16> %s) #2
  %add = add nsw i32 %call1, %call
  %call2 = tail call spir_func i32 @_Z3anyDv2_i(<2 x i32> %i) #2
  %add3 = add nsw i32 %add, %call2
  %call4 = tail call spir_func i32 @_Z3anyDv2_l(<2 x i64> %l) #2
  %add5 = add nsw i32 %add3, %call4
  %call6 = tail call spir_func i32 @_Z3allDv2_c(<2 x i8> %c) #2
  %add7 = add nsw i32 %add5, %call6
  %call8 = tail call spir_func i32 @_Z3allDv2_s(<2 x i16> %s) #2
  %add9 = add nsw i32 %add7, %call8
  %call10 = tail call spir_func i32 @_Z3allDv2_i(<2 x i32> %i) #2
  %add11 = add nsw i32 %add9, %call10
  %call12 = tail call spir_func i32 @_Z3allDv2_l(<2 x i64> %l) #2
  %add13 = add nsw i32 %add11, %call12
  store i32 %add13, i32 addrspace(4)* %out, align 4, !tbaa !3
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z3anyDv2_c(<2 x i8>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z3anyDv2_s(<2 x i16>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z3anyDv2_i(<2 x i32>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z3anyDv2_l(<2 x i64>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z3allDv2_c(<2 x i8>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z3allDv2_s(<2 x i16>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z3allDv2_i(<2 x i32>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z3allDv2_l(<2 x i64>) local_unnamed_addr #1

attributes #0 = { convergent mustprogress nofree norecurse nounwind willreturn writeonly "frame-pointer"="none" "min-legal-vector-width"="128" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent mustprogress nofree nounwind readnone willreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nounwind readnone willreturn }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 14.0.0"}
!3 = !{!4, !4, i64 0}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
