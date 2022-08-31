; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv -spirv-ext=+SPV_INTEL_variable_length_array
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck %s --input-file %t.rev.ll --check-prefix CHECK-LLVM

; CHECK-LLVM: phi i8* [ [[savedstack:%.*]], {{.*}} ], [ [[savedstack_us:%.*]], {{.*}} ]

; CHECK-LLVM: BB.{{[0-9]+}}:
; CHECK-LLVM: [[savedstack]] = call i8* @llvm.stacksave()

; CHECK-LLVM: BB.{{[0-9]+}}:
; CHECK-LLVM: [[savedstack_us]] = call i8* @llvm.stacksave()

; ModuleID = 's.bc'
source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

; Function Attrs: noinline nounwind mustprogress
define weak dso_local spir_kernel void @Kernel(i32 addrspace(1)* %0, i32 addrspace(1)* %1, i64 %.omp.lb.ascast.val109.zext, i64 %.omp.ub.ascast.val.zext, i64 %.capture_expr.0.ascast.val.zext, i64 %length_.ascast.val.zext) local_unnamed_addr #0 {
BB.0:
  %dmt.i = alloca [624 x i32], align 4
  %length_.ascast.val.zext.trunc = trunc i64 %length_.ascast.val.zext to i32
  %.capture_expr.0.ascast.val.zext.trunc = trunc i64 %.capture_expr.0.ascast.val.zext to i32
  %.omp.ub.ascast.val.zext.trunc = trunc i64 %.omp.ub.ascast.val.zext to i32
  %.omp.lb.ascast.val109.zext.trunc = trunc i64 %.omp.lb.ascast.val109.zext to i32
  %cmp41 = icmp slt i32 %.capture_expr.0.ascast.val.zext.trunc, 1
  %cmp42.not104 = icmp sgt i32 %.omp.lb.ascast.val109.zext.trunc, %.omp.ub.ascast.val.zext.trunc
  %or.cond = select i1 %cmp41, i1 true, i1 %cmp42.not104
  br i1 %or.cond, label %BB.2, label %BB.3

BB.1:                                             ; preds = %BB.12, %BB.7
  %savedstack.sink = phi i8* [ %savedstack, %BB.12 ], [ %savedstack.us, %BB.7 ]
  call void @llvm.stackrestore(i8* %savedstack.sink), !llvm.access.group !9
  br label %BB.2

BB.2:                                             ; preds = %BB.3, %BB.1, %BB.0
  ret void

BB.3:                                             ; preds = %BB.0
  %2 = call spir_func i64 @_Z13get_global_idj(i32 0)
  %3 = trunc i64 %2 to i32
  %.not = icmp sgt i32 %3, %.omp.ub.ascast.val.zext.trunc
  br i1 %.not, label %BB.2, label %BB.4, !prof !10

BB.4:                                             ; preds = %BB.3
  %div.i = udiv i32 %length_.ascast.val.zext.trunc, 624
  %4 = icmp ult i32 %length_.ascast.val.zext.trunc, 624
  br i1 %4, label %BB.5, label %BB.6

BB.5:                                             ; preds = %BB.4
  %savedstack = call i8* @llvm.stacksave(), !llvm.access.group !9
  br label %BB.12

BB.6:                                             ; preds = %BB.4
  %5 = icmp ugt i32 %div.i, 1
  %umax = select i1 %5, i32 %div.i, i32 1
  %arrayidx4812.us = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 %2
  %arrayidx5113.us = getelementptr inbounds i32, i32 addrspace(1)* %1, i64 %2
  %savedstack.us = call i8* @llvm.stacksave(), !llvm.access.group !9
  br label %BB.9

BB.7:                                             ; preds = %BB.8
  %indvars.iv.next27 = add nuw nsw i64 %indvars.iv26, 1
  %exitcond29.not = icmp eq i64 %indvars.iv.next27, %7
  br i1 %exitcond29.not, label %BB.1, label %BB.11, !llvm.loop !11

BB.8:                                             ; preds = %BB.11, %BB.8
  %indvars.iv22 = phi i64 [ 0, %BB.11 ], [ %indvars.iv.next23, %BB.8 ]
  %j.0.i17.us = phi i32 [ 0, %BB.11 ], [ %add.i.us, %BB.8 ]
  %arrayidx1276.i.us = getelementptr inbounds [624 x i32], [624 x i32]* %dmt.i, i64 0, i64 %indvars.iv22
  %6 = load i32, i32* %arrayidx1276.i.us, align 4, !alias.scope !14, !noalias !19, !llvm.access.group !9
  %and.i.us = and i32 %6, -2147483648
  %indvars.iv.next23 = add nuw nsw i64 %indvars.iv22, 1
  %add.i.us = add nuw nsw i32 %j.0.i17.us, 1
  %exitcond25.not = icmp eq i64 %indvars.iv.next23, 624
  br i1 %exitcond25.not, label %BB.7, label %BB.8, !llvm.loop !26

BB.9:                                             ; preds = %BB.9, %BB.6
  %indvars.iv = phi i64 [ %indvars.iv.next, %BB.9 ], [ 0, %BB.6 ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 624
  br i1 %exitcond.not, label %BB.10, label %BB.9, !llvm.loop !28

BB.10:                                            ; preds = %BB.9
  %7 = zext i32 %umax to i64
  br label %BB.11

BB.11:                                            ; preds = %BB.10, %BB.7
  %indvars.iv26 = phi i64 [ 0, %BB.10 ], [ %indvars.iv.next27, %BB.7 ]
  %8 = mul nuw nsw i64 %indvars.iv26, 624
  br label %BB.8

BB.12:                                            ; preds = %BB.12, %BB.5
  %indvars.iv33 = phi i64 [ 0, %BB.5 ], [ %indvars.iv.next34, %BB.12 ]
  %indvars.iv.next34 = add nuw nsw i64 %indvars.iv33, 1
  %exitcond35.not = icmp eq i64 %indvars.iv.next34, 624
  br i1 %exitcond35.not, label %BB.1, label %BB.12, !llvm.loop !11
}

declare spir_func i64 @_Z13get_global_idj(i32) local_unnamed_addr

; Function Attrs: nofree nosync nounwind willreturn
declare i8* @llvm.stacksave() #1

; Function Attrs: nofree nosync nounwind willreturn
declare void @llvm.stackrestore(i8*) #1

attributes #0 = { noinline nounwind mustprogress "contains-openmp-target"="true" "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" "frame-pointer"="all" "may-have-openmp-directive"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target.declare"="true" "unsafe-fp-math"="true" }
attributes #1 = { nofree nosync nounwind willreturn }

!opencl.used.extensions = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!opencl.used.optional.core.features = !{!1, !0, !0, !0, !1, !0, !1, !0, !1, !0, !1, !0, !0, !0}
!opencl.compiler.options = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!llvm.ident = !{!2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2}
!spirv.Source = !{!3, !4, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3}
!spirv.MemoryModel = !{!5}
!spirv.ExecutionMode = !{}
!llvm.module.flags = !{!6, !7, !8}
!sycl.specialization-constants = !{}

!0 = !{}
!1 = !{!"cl_doubles"}
!2 = !{!"Compiler"}
!3 = !{i32 4, i32 200000}
!4 = !{i32 3, i32 200000}
!5 = !{i32 2, i32 2}
!6 = !{i32 1, !"wchar_size", i32 4}
!7 = !{i32 7, !"PIC Level", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = distinct !{}
!10 = !{!"branch_weights", i32 100000, i32 299998}
!11 = distinct !{!11, !12, !13}
!12 = !{!"llvm.loop.vectorize.ivdep_loop", i32 0}
!13 = !{!"llvm.loop.parallel_accesses", !9}
!14 = !{!15, !17}
!15 = distinct !{!15, !16}
!16 = distinct !{!16}
!17 = distinct !{!17, !18}
!18 = distinct !{!18}
!19 = !{!20, !21, !22, !23, !24, !25}
!20 = distinct !{!20, !16}
!21 = distinct !{!21, !16}
!22 = distinct !{!22, !18}
!23 = distinct !{!23, !18}
!24 = distinct !{!24, !18}
!25 = distinct !{!25, !18}
!26 = distinct !{!26, !27}
!27 = !{!"llvm.loop.mustprogress"}
!28 = distinct !{!28, !27}
