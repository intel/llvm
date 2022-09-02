; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.spv.bc --spirv-target-env=SPV-IR
; RUN: llvm-dis < %t.rev.spv.bc | FileCheck %s --check-prefix=CHECK-SPV-IR
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; This test checks following SYCL relational builtins with half and half2 types:
;   isfinite, isinf, isnan, isnormal, signbit, isequal, isnotequal, isgreater
;   isgreaterequal, isless, islessequal, islessgreater, isordered, isunordered

; CHECK-SPIRV: 2 TypeBool [[BoolTypeID:[0-9]+]]
; CHECK-SPIRV: 4 TypeVector [[BoolVectorTypeID:[0-9]+]] [[BoolTypeID]] 2

; CHECK-SPIRV: 4 IsFinite [[BoolTypeID]]
; CHECK-SPIRV: 4 IsInf [[BoolTypeID]]
; CHECK-SPIRV: 4 IsNan [[BoolTypeID]]
; CHECK-SPIRV: 4 IsNormal [[BoolTypeID]]
; CHECK-SPIRV: 4 SignBitSet [[BoolTypeID]]
; CHECK-SPIRV: 5 FOrdEqual [[BoolTypeID]]
; CHECK-SPIRV: 5 FUnordNotEqual [[BoolTypeID]]
; CHECK-SPIRV: 5 FOrdGreaterThan [[BoolTypeID]]
; CHECK-SPIRV: 5 FOrdGreaterThanEqual [[BoolTypeID]]
; CHECK-SPIRV: 5 FOrdLessThan [[BoolTypeID]]
; CHECK-SPIRV: 5 FOrdLessThanEqual [[BoolTypeID]]
; CHECK-SPIRV: 5 FOrdNotEqual [[BoolTypeID]]
; CHECK-SPIRV: 5 Ordered [[BoolTypeID]]
; CHECK-SPIRV: 5 Unordered [[BoolTypeID]]

; CHECK-SPIRV: 4 IsFinite [[BoolVectorTypeID]]
; CHECK-SPIRV: 4 IsInf [[BoolVectorTypeID]]
; CHECK-SPIRV: 4 IsNan [[BoolVectorTypeID]]
; CHECK-SPIRV: 4 IsNormal [[BoolVectorTypeID]]
; CHECK-SPIRV: 4 SignBitSet [[BoolVectorTypeID]]
; CHECK-SPIRV: 5 FOrdEqual [[BoolVectorTypeID]]
; CHECK-SPIRV: 5 FUnordNotEqual [[BoolVectorTypeID]]
; CHECK-SPIRV: 5 FOrdGreaterThan [[BoolVectorTypeID]]
; CHECK-SPIRV: 5 FOrdGreaterThanEqual [[BoolVectorTypeID]]
; CHECK-SPIRV: 5 FOrdLessThan [[BoolVectorTypeID]]
; CHECK-SPIRV: 5 FOrdLessThanEqual [[BoolVectorTypeID]]
; CHECK-SPIRV: 5 FOrdNotEqual [[BoolVectorTypeID]]
; CHECK-SPIRV: 5 Ordered [[BoolVectorTypeID]]
; CHECK-SPIRV: 5 Unordered [[BoolVectorTypeID]]

; CHECK-SPV-IR: call spir_func i1 @_Z16__spirv_IsFiniteDh(half
; CHECK-SPV-IR: call spir_func i1 @_Z13__spirv_IsInfDh(half
; CHECK-SPV-IR: call spir_func i1 @_Z13__spirv_IsNanDh(half
; CHECK-SPV-IR: call spir_func i1 @_Z16__spirv_IsNormalDh(half
; CHECK-SPV-IR: call spir_func i1 @_Z18__spirv_SignBitSetDh(half
; CHECK-SPV-IR: fcmp oeq half
; CHECK-SPV-IR: fcmp une half
; CHECK-SPV-IR: fcmp ogt half
; CHECK-SPV-IR: fcmp oge half
; CHECK-SPV-IR: fcmp olt half
; CHECK-SPV-IR: fcmp ole half
; CHECK-SPV-IR: fcmp one half
; CHECK-SPV-IR: fcmp ord half
; CHECK-SPV-IR: fcmp uno half

; CHECK-SPV-IR: call spir_func <2 x i8> @_Z16__spirv_IsFiniteDv2_Dh(<2 x half>
; CHECK-SPV-IR: call spir_func <2 x i8> @_Z13__spirv_IsInfDv2_Dh(<2 x half>
; CHECK-SPV-IR: call spir_func <2 x i8> @_Z13__spirv_IsNanDv2_Dh(<2 x half>
; CHECK-SPV-IR: call spir_func <2 x i8> @_Z16__spirv_IsNormalDv2_Dh(<2 x half>
; CHECK-SPV-IR: call spir_func <2 x i8> @_Z18__spirv_SignBitSetDv2_Dh(<2 x half>
; CHECK-SPV-IR: fcmp oeq <2 x half>
; CHECK-SPV-IR: fcmp une <2 x half>
; CHECK-SPV-IR: fcmp ogt <2 x half>
; CHECK-SPV-IR: fcmp oge <2 x half>
; CHECK-SPV-IR: fcmp olt <2 x half>
; CHECK-SPV-IR: fcmp ole <2 x half>
; CHECK-SPV-IR: fcmp one <2 x half>
; CHECK-SPV-IR: fcmp ord <2 x half>
; CHECK-SPV-IR: fcmp uno <2 x half>

; CHECK-LLVM: call spir_func i32 @_Z8isfiniteDh(half
; CHECK-LLVM: call spir_func i32 @_Z5isinfDh(half
; CHECK-LLVM: call spir_func i32 @_Z5isnanDh(half
; CHECK-LLVM: call spir_func i32 @_Z8isnormalDh(half
; CHECK-LLVM: call spir_func i32 @_Z7signbitDh(half
; CHECK-LLVM: fcmp oeq half
; CHECK-LLVM: fcmp une half
; CHECK-LLVM: fcmp ogt half
; CHECK-LLVM: fcmp oge half
; CHECK-LLVM: fcmp olt half
; CHECK-LLVM: fcmp ole half
; CHECK-LLVM: fcmp one half
; CHECK-LLVM: fcmp ord half
; CHECK-LLVM: fcmp uno half

; CHECK-LLVM: call spir_func <2 x i16> @_Z8isfiniteDv2_Dh(<2 x half>
; CHECK-LLVM: call spir_func <2 x i16> @_Z5isinfDv2_Dh(<2 x half>
; CHECK-LLVM: call spir_func <2 x i16> @_Z5isnanDv2_Dh(<2 x half>
; CHECK-LLVM: call spir_func <2 x i16> @_Z8isnormalDv2_Dh(<2 x half>
; CHECK-LLVM: call spir_func <2 x i16> @_Z7signbitDv2_Dh(<2 x half>
; CHECK-LLVM: fcmp oeq <2 x half>
; CHECK-LLVM: fcmp une <2 x half>
; CHECK-LLVM: fcmp ogt <2 x half>
; CHECK-LLVM: fcmp oge <2 x half>
; CHECK-LLVM: fcmp olt <2 x half>
; CHECK-LLVM: fcmp ole <2 x half>
; CHECK-LLVM: fcmp one <2 x half>
; CHECK-LLVM: fcmp ord <2 x half>
; CHECK-LLVM: fcmp uno <2 x half>

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn writeonly
define dso_local spir_func void @test_scalar(i32 addrspace(4)* nocapture writeonly %out, half %h) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func i32 @_Z8isfiniteDh(half %h) #3
  %call1 = tail call spir_func i32 @_Z5isinfDh(half %h) #3
  %add = add nsw i32 %call1, %call
  %call2 = tail call spir_func i32 @_Z5isnanDh(half %h) #3
  %add3 = add nsw i32 %add, %call2
  %call4 = tail call spir_func i32 @_Z8isnormalDh(half %h) #3
  %add5 = add nsw i32 %add3, %call4
  %call6 = tail call spir_func i32 @_Z7signbitDh(half %h) #3
  %add7 = add nsw i32 %add5, %call6
  %call8 = tail call spir_func i32 @_Z7isequalDhDh(half %h, half %h) #3
  %add9 = add nsw i32 %add7, %call8
  %call10 = tail call spir_func i32 @_Z10isnotequalDhDh(half %h, half %h) #3
  %add11 = add nsw i32 %add9, %call10
  %call12 = tail call spir_func i32 @_Z9isgreaterDhDh(half %h, half %h) #3
  %add13 = add nsw i32 %add11, %call12
  %call14 = tail call spir_func i32 @_Z14isgreaterequalDhDh(half %h, half %h) #3
  %add15 = add nsw i32 %add13, %call14
  %call16 = tail call spir_func i32 @_Z6islessDhDh(half %h, half %h) #3
  %add17 = add nsw i32 %add15, %call16
  %call18 = tail call spir_func i32 @_Z11islessequalDhDh(half %h, half %h) #3
  %add19 = add nsw i32 %add17, %call18
  %call20 = tail call spir_func i32 @_Z13islessgreaterDhDh(half %h, half %h) #3
  %add21 = add nsw i32 %add19, %call20
  %call22 = tail call spir_func i32 @_Z9isorderedDhDh(half %h, half %h) #3
  %add23 = add nsw i32 %add21, %call22
  %call24 = tail call spir_func i32 @_Z11isunorderedDhDh(half %h, half %h) #3
  %add25 = add nsw i32 %add23, %call24
  store i32 %add25, i32 addrspace(4)* %out, align 4, !tbaa !3
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z8isfiniteDh(half) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z5isinfDh(half) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z5isnanDh(half) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z8isnormalDh(half) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z7signbitDh(half) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z7isequalDhDh(half, half) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z10isnotequalDhDh(half, half) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z9isgreaterDhDh(half, half) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z14isgreaterequalDhDh(half, half) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z6islessDhDh(half, half) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z11islessequalDhDh(half, half) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z13islessgreaterDhDh(half, half) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z9isorderedDhDh(half, half) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z11isunorderedDhDh(half, half) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn writeonly
define dso_local spir_func void @test_vector(<2 x i16> addrspace(4)* nocapture writeonly %out, <2 x half> %h) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func <2 x i16> @_Z8isfiniteDv2_Dh(<2 x half> %h) #3
  %call1 = tail call spir_func <2 x i16> @_Z5isinfDv2_Dh(<2 x half> %h) #3
  %add = add <2 x i16> %call1, %call
  %call2 = tail call spir_func <2 x i16> @_Z5isnanDv2_Dh(<2 x half> %h) #3
  %add3 = add <2 x i16> %add, %call2
  %call4 = tail call spir_func <2 x i16> @_Z8isnormalDv2_Dh(<2 x half> %h) #3
  %add5 = add <2 x i16> %add3, %call4
  %call6 = tail call spir_func <2 x i16> @_Z7signbitDv2_Dh(<2 x half> %h) #3
  %add7 = add <2 x i16> %add5, %call6
  %call8 = tail call spir_func <2 x i16> @_Z7isequalDv2_DhS_(<2 x half> %h, <2 x half> %h) #3
  %add9 = add <2 x i16> %add7, %call8
  %call10 = tail call spir_func <2 x i16> @_Z10isnotequalDv2_DhS_(<2 x half> %h, <2 x half> %h) #3
  %add11 = add <2 x i16> %add9, %call10
  %call12 = tail call spir_func <2 x i16> @_Z9isgreaterDv2_DhS_(<2 x half> %h, <2 x half> %h) #3
  %add13 = add <2 x i16> %add11, %call12
  %call14 = tail call spir_func <2 x i16> @_Z14isgreaterequalDv2_DhS_(<2 x half> %h, <2 x half> %h) #3
  %add15 = add <2 x i16> %add13, %call14
  %call16 = tail call spir_func <2 x i16> @_Z6islessDv2_DhS_(<2 x half> %h, <2 x half> %h) #3
  %add17 = add <2 x i16> %add15, %call16
  %call18 = tail call spir_func <2 x i16> @_Z11islessequalDv2_DhS_(<2 x half> %h, <2 x half> %h) #3
  %add19 = add <2 x i16> %add17, %call18
  %call20 = tail call spir_func <2 x i16> @_Z13islessgreaterDv2_DhS_(<2 x half> %h, <2 x half> %h) #3
  %add21 = add <2 x i16> %add19, %call20
  %call22 = tail call spir_func <2 x i16> @_Z9isorderedDv2_DhS_(<2 x half> %h, <2 x half> %h) #3
  %add23 = add <2 x i16> %add21, %call22
  %call24 = tail call spir_func <2 x i16> @_Z11isunorderedDv2_DhS_(<2 x half> %h, <2 x half> %h) #3
  %add25 = add <2 x i16> %add23, %call24
  store <2 x i16> %add25, <2 x i16> addrspace(4)* %out, align 4, !tbaa !7
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i16> @_Z8isfiniteDv2_Dh(<2 x half>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i16> @_Z5isinfDv2_Dh(<2 x half>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i16> @_Z5isnanDv2_Dh(<2 x half>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i16> @_Z8isnormalDv2_Dh(<2 x half>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i16> @_Z7signbitDv2_Dh(<2 x half>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i16> @_Z7isequalDv2_DhS_(<2 x half>, <2 x half>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i16> @_Z10isnotequalDv2_DhS_(<2 x half>, <2 x half>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i16> @_Z9isgreaterDv2_DhS_(<2 x half>, <2 x half>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i16> @_Z14isgreaterequalDv2_DhS_(<2 x half>, <2 x half>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i16> @_Z6islessDv2_DhS_(<2 x half>, <2 x half>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i16> @_Z11islessequalDv2_DhS_(<2 x half>, <2 x half>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i16> @_Z13islessgreaterDv2_DhS_(<2 x half>, <2 x half>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i16> @_Z9isorderedDv2_DhS_(<2 x half>, <2 x half>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i16> @_Z11isunorderedDv2_DhS_(<2 x half>, <2 x half>) local_unnamed_addr #1

attributes #0 = { convergent mustprogress nofree norecurse nounwind willreturn writeonly "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent mustprogress nofree nounwind readnone willreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent mustprogress nofree norecurse nounwind willreturn writeonly "frame-pointer"="none" "min-legal-vector-width"="32" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent nounwind readnone willreturn }

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
!7 = !{!5, !5, i64 0}
