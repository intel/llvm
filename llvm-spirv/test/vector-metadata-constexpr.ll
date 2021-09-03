; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -s %t.bc -o - | llvm-dis -o - | FileCheck %s
; RUN: llvm-spirv %t.bc

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind ssp uwtable
define void @foo() #0 !dbg !4 {
entry:
  call void @llvm.dbg.value(metadata <3 x i8> <i8 add (i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 65793, i32 65793> to <8 x i8>), i32 0), i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 131586, i32 131586> to <8 x i8>), i32 0)), i8 add (i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 65793, i32 65793> to <8 x i8>), i32 1), i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 131586, i32 131586> to <8 x i8>), i32 1)), i8 add (i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 65793, i32 65793> to <8 x i8>), i32 2), i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 131586, i32 131586> to <8 x i8>), i32 2))>, metadata !12, metadata !DIExpression()), !dbg !18
  ret void, !dbg !20
; CHECK: %0 = bitcast <2 x i32> <i32 65793, i32 65793> to <8 x i8>
; CHECK: %1 = extractelement <8 x i8> %0, i32 0
; CHECK: %2 = bitcast <2 x i32> <i32 131586, i32 131586> to <8 x i8>
; CHECK: %3 = extractelement <8 x i8> %2, i32 0
; CHECK: %4 = add i8 %1, %3
; CHECK: %5 = bitcast <2 x i32> <i32 65793, i32 65793> to <8 x i8>
; CHECK: %6 = extractelement <8 x i8> %5, i32 1
; CHECK: %7 = bitcast <2 x i32> <i32 131586, i32 131586> to <8 x i8>
; CHECK: %8 = extractelement <8 x i8> %7, i32 1
; CHECK: %9 = add i8 %6, %8
; CHECK: %10 = bitcast <2 x i32> <i32 65793, i32 65793> to <8 x i8>
; CHECK: %11 = extractelement <8 x i8> %10, i32 2
; CHECK: %12 = bitcast <2 x i32> <i32 131586, i32 131586> to <8 x i8>
; CHECK: %13 = extractelement <8 x i8> %12, i32 2
; CHECK: %14 = add i8 %11, %13
; CHECK: %15 = insertelement <3 x i8> undef, i8 %4, i32 0
; CHECK: %16 = insertelement <3 x i8> %15, i8 %9, i32 1
; CHECK: %17 = insertelement <3 x i8> %16, i8 %14, i32 2
; CHECK: call void @llvm.dbg.value(metadata <3 x i8> %17, metadata !{{[0-9]+}}, metadata !DIExpression()), !dbg !{{[0-9]+}}
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind ssp uwtable  }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14, !15}
!llvm.ident = !{!16}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.7.0 (trunk 235110) (llvm/trunk 235108)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "t.c", directory: "/path/to/dir")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 2, type: !8, isLocal: true, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !1, line: 2, type: !10)
!13 = !{i32 2, !"Dwarf Version", i32 2}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"PIC Level", i32 2}
!16 = !{!"clang version 3.7.0 (trunk 235110) (llvm/trunk 235108)"}
!17 = !DIExpression()
!18 = !DILocation(line: 2, column: 52, scope: !7, inlinedAt: !19)
!19 = distinct !DILocation(line: 4, column: 3, scope: !4)
!20 = !DILocation(line: 6, column: 1, scope: !4)
