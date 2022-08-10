; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.txt
; RUN: FileCheck < %t.txt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.spv.bc --spirv-target-env=SPV-IR
; RUN: llvm-dis < %t.rev.spv.bc | FileCheck %s --check-prefix=CHECK-SPV-IR
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; This test checks following SYCL relational builtins with double and double2
; types:
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

; CHECK-SPV-IR: call spir_func i1 @_Z16__spirv_IsFinited(double
; CHECK-SPV-IR: call spir_func i1 @_Z13__spirv_IsInfd(double
; CHECK-SPV-IR: call spir_func i1 @_Z13__spirv_IsNand(double
; CHECK-SPV-IR: call spir_func i1 @_Z16__spirv_IsNormald(double
; CHECK-SPV-IR: call spir_func i1 @_Z18__spirv_SignBitSetd(double
; CHECK-SPV-IR: fcmp oeq double
; CHECK-SPV-IR: fcmp une double
; CHECK-SPV-IR: fcmp ogt double
; CHECK-SPV-IR: fcmp oge double
; CHECK-SPV-IR: fcmp olt double
; CHECK-SPV-IR: fcmp ole double
; CHECK-SPV-IR: fcmp one double
; CHECK-SPV-IR: fcmp ord double
; CHECK-SPV-IR: fcmp uno double

; CHECK-SPV-IR: call spir_func <2 x i8> @_Z16__spirv_IsFiniteDv2_d(<2 x double>
; CHECK-SPV-IR: call spir_func <2 x i8> @_Z13__spirv_IsInfDv2_d(<2 x double>
; CHECK-SPV-IR: call spir_func <2 x i8> @_Z13__spirv_IsNanDv2_d(<2 x double>
; CHECK-SPV-IR: call spir_func <2 x i8> @_Z16__spirv_IsNormalDv2_d(<2 x double>
; CHECK-SPV-IR: call spir_func <2 x i8> @_Z18__spirv_SignBitSetDv2_d(<2 x double>
; CHECK-SPV-IR: fcmp oeq <2 x double>
; CHECK-SPV-IR: fcmp une <2 x double>
; CHECK-SPV-IR: fcmp ogt <2 x double>
; CHECK-SPV-IR: fcmp oge <2 x double>
; CHECK-SPV-IR: fcmp olt <2 x double>
; CHECK-SPV-IR: fcmp ole <2 x double>
; CHECK-SPV-IR: fcmp one <2 x double>
; CHECK-SPV-IR: fcmp ord <2 x double>
; CHECK-SPV-IR: fcmp uno <2 x double>

; CHECK-LLVM: call spir_func i32 @_Z8isfinited(double
; CHECK-LLVM: call spir_func i32 @_Z5isinfd(double
; CHECK-LLVM: call spir_func i32 @_Z5isnand(double
; CHECK-LLVM: call spir_func i32 @_Z8isnormald(double
; CHECK-LLVM: call spir_func i32 @_Z7signbitd(double
; CHECK-LLVM: fcmp oeq double
; CHECK-LLVM: fcmp une double
; CHECK-LLVM: fcmp ogt double
; CHECK-LLVM: fcmp oge double
; CHECK-LLVM: fcmp olt double
; CHECK-LLVM: fcmp ole double
; CHECK-LLVM: fcmp one double
; CHECK-LLVM: fcmp ord double
; CHECK-LLVM: fcmp uno double

; CHECK-LLVM: call spir_func <2 x i64> @_Z8isfiniteDv2_d(<2 x double>
; CHECK-LLVM: call spir_func <2 x i64> @_Z5isinfDv2_d(<2 x double>
; CHECK-LLVM: call spir_func <2 x i64> @_Z5isnanDv2_d(<2 x double>
; CHECK-LLVM: call spir_func <2 x i64> @_Z8isnormalDv2_d(<2 x double>
; CHECK-LLVM: call spir_func <2 x i64> @_Z7signbitDv2_d(<2 x double>
; CHECK-LLVM: fcmp oeq <2 x double>
; CHECK-LLVM: fcmp une <2 x double>
; CHECK-LLVM: fcmp ogt <2 x double>
; CHECK-LLVM: fcmp oge <2 x double>
; CHECK-LLVM: fcmp olt <2 x double>
; CHECK-LLVM: fcmp ole <2 x double>
; CHECK-LLVM: fcmp one <2 x double>
; CHECK-LLVM: fcmp ord <2 x double>
; CHECK-LLVM: fcmp uno <2 x double>

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn writeonly
define dso_local spir_func void @test_scalar(i32 addrspace(4)* nocapture writeonly %out, double %d) local_unnamed_addr #0 {
entry:
  %call = tail call spir_func i32 @_Z8isfinited(double %d) #3
  %call1 = tail call spir_func i32 @_Z5isinfd(double %d) #3
  %add = add nsw i32 %call1, %call
  %call2 = tail call spir_func i32 @_Z5isnand(double %d) #3
  %add3 = add nsw i32 %add, %call2
  %call4 = tail call spir_func i32 @_Z8isnormald(double %d) #3
  %add5 = add nsw i32 %add3, %call4
  %call6 = tail call spir_func i32 @_Z7signbitd(double %d) #3
  %add7 = add nsw i32 %add5, %call6
  %call8 = tail call spir_func i32 @_Z7isequaldd(double %d, double %d) #3
  %add9 = add nsw i32 %add7, %call8
  %call10 = tail call spir_func i32 @_Z10isnotequaldd(double %d, double %d) #3
  %add11 = add nsw i32 %add9, %call10
  %call12 = tail call spir_func i32 @_Z9isgreaterdd(double %d, double %d) #3
  %add13 = add nsw i32 %add11, %call12
  %call14 = tail call spir_func i32 @_Z14isgreaterequaldd(double %d, double %d) #3
  %add15 = add nsw i32 %add13, %call14
  %call16 = tail call spir_func i32 @_Z6islessdd(double %d, double %d) #3
  %add17 = add nsw i32 %add15, %call16
  %call18 = tail call spir_func i32 @_Z11islessequaldd(double %d, double %d) #3
  %add19 = add nsw i32 %add17, %call18
  %call20 = tail call spir_func i32 @_Z13islessgreaterdd(double %d, double %d) #3
  %add21 = add nsw i32 %add19, %call20
  %call22 = tail call spir_func i32 @_Z9isordereddd(double %d, double %d) #3
  %add23 = add nsw i32 %add21, %call22
  %call24 = tail call spir_func i32 @_Z11isunordereddd(double %d, double %d) #3
  %add25 = add nsw i32 %add23, %call24
  store i32 %add25, i32 addrspace(4)* %out, align 4, !tbaa !3
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z8isfinited(double) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z5isinfd(double) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z5isnand(double) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z8isnormald(double) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z7signbitd(double) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z7isequaldd(double, double) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z10isnotequaldd(double, double) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z9isgreaterdd(double, double) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z14isgreaterequaldd(double, double) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z6islessdd(double, double) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z11islessequaldd(double, double) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z13islessgreaterdd(double, double) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z9isordereddd(double, double) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @_Z11isunordereddd(double, double) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn writeonly
define dso_local spir_func void @test_vector(<2 x i64> addrspace(4)* nocapture writeonly %out, <2 x double> %d) local_unnamed_addr #2 {
entry:
  %call = tail call spir_func <2 x i64> @_Z8isfiniteDv2_d(<2 x double> %d) #3
  %call1 = tail call spir_func <2 x i64> @_Z5isinfDv2_d(<2 x double> %d) #3
  %add = add <2 x i64> %call1, %call
  %call2 = tail call spir_func <2 x i64> @_Z5isnanDv2_d(<2 x double> %d) #3
  %add3 = add <2 x i64> %add, %call2
  %call4 = tail call spir_func <2 x i64> @_Z8isnormalDv2_d(<2 x double> %d) #3
  %add5 = add <2 x i64> %add3, %call4
  %call6 = tail call spir_func <2 x i64> @_Z7signbitDv2_d(<2 x double> %d) #3
  %add7 = add <2 x i64> %add5, %call6
  %call8 = tail call spir_func <2 x i64> @_Z7isequalDv2_dS_(<2 x double> %d, <2 x double> %d) #3
  %add9 = add <2 x i64> %add7, %call8
  %call10 = tail call spir_func <2 x i64> @_Z10isnotequalDv2_dS_(<2 x double> %d, <2 x double> %d) #3
  %add11 = add <2 x i64> %add9, %call10
  %call12 = tail call spir_func <2 x i64> @_Z9isgreaterDv2_dS_(<2 x double> %d, <2 x double> %d) #3
  %add13 = add <2 x i64> %add11, %call12
  %call14 = tail call spir_func <2 x i64> @_Z14isgreaterequalDv2_dS_(<2 x double> %d, <2 x double> %d) #3
  %add15 = add <2 x i64> %add13, %call14
  %call16 = tail call spir_func <2 x i64> @_Z6islessDv2_dS_(<2 x double> %d, <2 x double> %d) #3
  %add17 = add <2 x i64> %add15, %call16
  %call18 = tail call spir_func <2 x i64> @_Z11islessequalDv2_dS_(<2 x double> %d, <2 x double> %d) #3
  %add19 = add <2 x i64> %add17, %call18
  %call20 = tail call spir_func <2 x i64> @_Z13islessgreaterDv2_dS_(<2 x double> %d, <2 x double> %d) #3
  %add21 = add <2 x i64> %add19, %call20
  %call22 = tail call spir_func <2 x i64> @_Z9isorderedDv2_dS_(<2 x double> %d, <2 x double> %d) #3
  %add23 = add <2 x i64> %add21, %call22
  %call24 = tail call spir_func <2 x i64> @_Z11isunorderedDv2_dS_(<2 x double> %d, <2 x double> %d) #3
  %add25 = add <2 x i64> %add23, %call24
  store <2 x i64> %add25, <2 x i64> addrspace(4)* %out, align 16, !tbaa !7
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i64> @_Z8isfiniteDv2_d(<2 x double>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i64> @_Z5isinfDv2_d(<2 x double>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i64> @_Z5isnanDv2_d(<2 x double>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i64> @_Z8isnormalDv2_d(<2 x double>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i64> @_Z7signbitDv2_d(<2 x double>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i64> @_Z7isequalDv2_dS_(<2 x double>, <2 x double>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i64> @_Z10isnotequalDv2_dS_(<2 x double>, <2 x double>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i64> @_Z9isgreaterDv2_dS_(<2 x double>, <2 x double>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i64> @_Z14isgreaterequalDv2_dS_(<2 x double>, <2 x double>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i64> @_Z6islessDv2_dS_(<2 x double>, <2 x double>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i64> @_Z11islessequalDv2_dS_(<2 x double>, <2 x double>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i64> @_Z13islessgreaterDv2_dS_(<2 x double>, <2 x double>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i64> @_Z9isorderedDv2_dS_(<2 x double>, <2 x double>) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func <2 x i64> @_Z11isunorderedDv2_dS_(<2 x double>, <2 x double>) local_unnamed_addr #1

attributes #0 = { convergent mustprogress nofree norecurse nounwind willreturn writeonly "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent mustprogress nofree nounwind readnone willreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent mustprogress nofree norecurse nounwind willreturn writeonly "frame-pointer"="none" "min-legal-vector-width"="128" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
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
