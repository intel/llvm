; RUN: llvm-spirv -s %s -o - | llvm-dis -o - | FileCheck %s

target triple = "spirv64"

; Preserve the bitcast's location on both halves and salvage its debug value
; using the original operand instead of replacing it with poison.
; CHECK-LABEL: define spir_func i8 @bitcast_debug(
; CHECK: bitcast <4 x i32> {{.*}} to <16 x i8>, !dbg ![[LOC:[0-9]+]]
; CHECK: bitcast <4 x i32> {{.*}} to <16 x i8>, !dbg ![[LOC]]
; CHECK: #dbg_value(<8 x i32> %v, ![[VAR:[0-9]+]], !DIExpression(), ![[LOC]])
define spir_func i8 @bitcast_debug(<8 x i32> %v) !dbg !3 {
  %bc = bitcast <8 x i32> %v to <32 x i8>, !dbg !9
    #dbg_value(<32 x i8> %bc, !6, !DIExpression(), !9)
  %lane = extractelement <32 x i8> %bc, i64 20
  ret i8 %lane
}

; CHECK-DAG: ![[VAR]] = !DILocalVariable(name: "bytes", scope: ![[SCOPE:[0-9]+]], file: !{{[0-9]+}}, line: 2, type: !{{[0-9]+}})
; CHECK-DAG: ![[LOC]] = !DILocation(line: 2, column: 3, scope: ![[SCOPE]])

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "bitcast.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "bitcast_debug", scope: !1, file: !1, line: 1, type: !4, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{!10}
!6 = !DILocalVariable(name: "bytes", scope: !3, file: !1, line: 2, type: !7)
!7 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 256, elements: !8)
!8 = !{!11}
!9 = !DILocation(line: 2, column: 3, scope: !3)
!10 = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
!11 = !DISubrange(count: 32)
