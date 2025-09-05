; This test verifies that there are no duplicates of any DebugLine extended instructions.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-debug-info-version=nonsemantic-shader-200
; RUN: spirv-val %t.spv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define spir_func void @mod_jtype1_mp_gpu_genr70_jtype1_(i64 %a4) local_unnamed_addr #0 !dbg !3336 {
do.end_do144:
  %add.35 = add nsw i64 %a4, 1, !dbg !3689
  tail call void @llvm.dbg.value(metadata i64 %add.35, metadata !3406, metadata !DIExpression()) #56, !dbg !3447
  %rel.27.not.not = icmp slt i64 %add.35, %a4, !dbg !3689
  br i1 %rel.27.not.not, label %do.end_do115, label %bb1, !dbg !3689, !llvm.loop !3690

bb1:                                              ; preds = %do.end_do144
  %add.36 = add nsw i64 %a4, 1, !dbg !3579
  tail call void @llvm.dbg.value(metadata i64 %add.36, metadata !3409, metadata !DIExpression()) #56, !dbg !3447
  %rel.28.not.not = icmp slt i64 %add.36, %a4, !dbg !3579
  br i1 %rel.28.not.not, label %do.end_do115, label %do.end_do119, !dbg !3579, !llvm.loop !3691

do.end_do119:                                     ; preds = %bb1, %do.end_do119
  %add.37 = add nsw i64 %a4, 1, !dbg !3692
  tail call void @llvm.dbg.value(metadata i64 %add.37, metadata !3411, metadata !DIExpression()) #56, !dbg !3447
  %rel.29.not.not = icmp slt i64 %add.37, %a4, !dbg !3692
  br i1 %rel.29.not.not, label %do.end_do119, label %do.end_do115, !dbg !3692, !llvm.loop !3693

do.end_do115:                                     ; preds = do.end_do144, bb1, %do.end_do119
  ret void, !dbg !3694
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "denormal-fp-math"="preserve-sign" "frame-pointer"="none" "intel-lang"="fortran" "may-have-openmp-directive"="false" "min-legal-vector-width"="0" "openmp-target-declare"="true" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!51}
!spirv.Source = !{!3325}
!llvm.module.flags = !{!3329}

!6 = !DISubroutineType(types: !7)
!7 = !{null}
!15 = !DIBasicType(name: "INTEGER*8", size: 64, encoding: DW_ATE_signed)

!51 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !52, producer: "Intel(R) Fortran 24.0-1652", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!52 = !DIFile(filename: "gpu_jtype1.F90", directory: "/tmp/bug")

!3325 = !{i32 4, i32 200000}
!3329 = !{i32 2, !"Debug Info Version", i32 3}

!3336 = distinct !DISubprogram(name: "gpu_genr70_jtype1", linkageName: "mod_jtype1_mp_gpu_genr70_jtype1_", scope: !3337, file: !52, line: 5, type: !6, scopeLine: 5, spFlags: DISPFlagDefinition, unit: !51)
!3337 = !DIModule(scope: null, name: "mod_jtype1", file: !52, line: 2)
!3406 = !DILocalVariable(name: "k", scope: !3336, file: !52, line: 40, type: !15)
!3409 = !DILocalVariable(name: "jj", scope: !3336, file: !52, line: 40, type: !15)
!3411 = !DILocalVariable(name: "ii", scope: !3336, file: !52, line: 40, type: !15)
!3447 = !DILocation(line: 0, scope: !3336)
!3578 = !DILocation(line: 182, column: 2, scope: !3336)
!3579 = !DILocation(line: 183, column: 2, scope: !3336)
!3583 = !DILocation(line: 227, column: 2, scope: !3336)
!3689 = !DILocation(line: 296, column: 7, scope: !3336)
!3690 = distinct !{!3690, !3583, !3689}
!3691 = distinct !{!3691, !3579, !3579}
!3692 = !DILocation(line: 298, column: 7, scope: !3336)
!3693 = distinct !{!3693, !3578, !3692}
!3694 = !DILocation(line: 306, column: 7, scope: !3336)
