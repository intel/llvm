; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s

source_filename = "the_file.ll"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_func i1 @trunc_to_i1(i32 %iarg) #0 !dbg !7 {
; CHECK: @trunc_to_i1(i32 %iarg) #[[#]] !dbg ![[#]] {
; CHECK-NEXT: !dbg ![[#TRUNC_LINE:]]
; CHECK-NEXT: !dbg ![[#TRUNC_LINE]]
; CHECK-NEXT: ret i1 %res, !dbg ![[#TRUNC_RET_LINE:]]
  %res = trunc i32 %iarg to i1, !dbg !9
  ret i1 %res, !dbg !10
}

; Function Attrs: nounwind
define spir_func i32 @sext_from_i1(i1 %barg) #0 !dbg !11 {
; CHECK: @sext_from_i1(i1 %barg) #[[#]] !dbg ![[#]] {
; CHECK-NEXT: !dbg ![[#SEXT_LINE:]]
; CHECK: ret i32 %res, !dbg ![[#SEXT_RET_LINE:]]
  %res = sext i1 %barg to i32, !dbg !12
  ret i32 %res, !dbg !13
}

; Function Attrs: nounwind
define spir_func i32 @zext_from_i1(i1 %barg) #0 !dbg !14 {
; CHECK: @zext_from_i1(i1 %barg) #[[#]] !dbg ![[#]] {
; CHECK-NEXT: !dbg ![[#ZEXT_LINE:]]
; CHECK: ret i32 %res, !dbg ![[#ZEXT_RET_LINE:]]
  %res = zext i1 %barg to i32, !dbg !15
  ret i32 %res, !dbg !16
}

; Function Attrs: nounwind
define spir_func float @sitofp_b(i1 %barg) #0 !dbg !17 {
; CHECK: @sitofp_b(i1 %barg) #[[#]] !dbg ![[#]] {
; CHECK-NEXT: !dbg ![[#SITOFP_LINE:]]
; CHECK-NEXT: !dbg ![[#SITOFP_LINE]]
; CHECK: ret float %res, !dbg ![[#SITOFP_RET_LINE:]]
  %res = sitofp i1 %barg to float, !dbg !18
  ret float %res, !dbg !19
}

; Function Attrs: nounwind
define spir_func float @uitofp_b(i1 %barg) #0 !dbg !20 {
; CHECK: @uitofp_b(i1 %barg) #[[#]] !dbg ![[#]] {
; CHECK-NEXT: !dbg ![[#UITOFP_LINE:]]
; CHECK-NEXT: !dbg ![[#UITOFP_LINE]]
; CHECK: ret float %res, !dbg ![[#UITOFP_RET_LINE:]]
  %res = uitofp i1 %barg to float, !dbg !21
  ret float %res, !dbg !22
}

; CHECK-DAG: ![[#TRUNC_LINE]] = !DILocation(line: 1, column: 1
; CHECK-DAG: ![[#TRUNC_RET_LINE]] = !DILocation(line: 2, column: 1

; CHECK-DAG: ![[#SEXT_LINE]] = !DILocation(line: 3, column: 1
; CHECK-DAG: ![[#SEXT_RET_LINE]] = !DILocation(line: 4, column: 1

; CHECK-DAG: ![[#ZEXT_LINE]] = !DILocation(line: 5, column: 1
; CHECK-DAG: ![[#ZEXT_RET_LINE]] = !DILocation(line: 6, column: 1

; CHECK-DAG: ![[#SITOFP_LINE]] = !DILocation(line: 7, column: 1
; CHECK-DAG: ![[#SITOFP_RET_LINE]] = !DILocation(line: 8, column: 1

; CHECK-DAG: ![[#UITOFP_LINE]] = !DILocation(line: 9, column: 1
; CHECK-DAG: ![[#UITOFP_RET_LINE]] = !DILocation(line: 10, column: 1

attributes #0 = { nounwind }

!llvm.dbg.cu = !{!2}
!llvm.debugify = !{!4, !5}
!llvm.module.flags = !{!6}

!0 = !{i32 1, i32 2}
!1 = !{}
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!3 = !DIFile(filename: "the_file.ll", directory: "", checksumkind: CSK_MD5, checksum: "18aa9ce738eaafc7b7b7181c19092815")
!4 = !{i32 10}
!5 = !{i32 0}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = distinct !DISubprogram(name: "trunc_to_i1", scope: !3, file: !3, line: 1, type: !8, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1)
!8 = !DISubroutineType(types: !1)
!9 = !DILocation(line: 1, column: 1, scope: !7)
!10 = !DILocation(line: 2, column: 1, scope: !7)
!11 = distinct !DISubprogram(name: "sext_from_i1", scope: !3, file: !3, line: 3, type: !8, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1)
!12 = !DILocation(line: 3, column: 1, scope: !11)
!13 = !DILocation(line: 4, column: 1, scope: !11)
!14 = distinct !DISubprogram(name: "zext_from_i1", scope: !3, file: !3, line: 5, type: !8, scopeLine: 5, flags: DIFlagPrototyped , spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1)
!15 = !DILocation(line: 5, column: 1, scope: !14)
!16 = !DILocation(line: 6, column: 1, scope: !14)
!17 = distinct !DISubprogram(name: "sitofp_b", scope: !3, file: !3, line: 7, type: !8, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1)
!18 = !DILocation(line: 7, column: 1, scope: !17)
!19 = !DILocation(line: 8, column: 1, scope: !17)
!20 = distinct !DISubprogram(name: "uitofp_b", scope: !3, file: !3, line: 9, type: !8, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !1)
!21 = !DILocation(line: 9, column: 1, scope: !20)
!22 = !DILocation(line: 10, column: 1, scope: !20)
