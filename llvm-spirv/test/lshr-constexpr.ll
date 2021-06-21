; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM
; RUN: llvm-spirv --to-text %t.spv -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; TODO: spirv-val command is not presented here because it fails with error:
; line 29: Invalid source language operand: 12 which is not related to
; functionality that is being checked in this test.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV: TypeInt [[type_int32:[0-9]+]] 32 0
; CHECK-SPIRV: TypeInt [[type_int64:[0-9]+]] 64 0
; CHECK-SPIRV: Constant [[type_int32]] [[const1:[0-9]+]] 1
; CHECK-SPIRV: Constant [[type_int64]] [[const32:[0-9]+]] 32 0
; CHECK-SPIRV: TypeVector [[type_vec:[0-9]+]] [[type_int32]] 2
; CHECK-SPIRV: ConstantComposite [[type_vec]] [[vec_const:[0-9]+]] [[const1]] [[const1]]

; CHECK-SPIRV: Bitcast [[type_int64]] [[bitcast_res:[0-9]+]] [[vec_const]]
; CHECK-SPIRV: ShiftRightLogical [[type_int64]] [[shift_res:[0-9]+]] [[bitcast_res]] [[const32]]
; CHECK-SPIRV: DebugValue {{[0-9]+}} [[shift_res]]

; Function Attrs: nounwind ssp uwtable
define void @foo() #0 !dbg !4 {
entry:
  tail call void @llvm.dbg.value(metadata i64 lshr (i64 bitcast (<2 x i32> <i32 1, i32 1> to i64), i64 32), metadata !12, metadata !17) #3, !dbg !18
  ret void, !dbg !20
; CHECK-LLVM: %[[bitcast:[0-9]+]] = bitcast <2 x i32> <i32 1, i32 1> to i64
; CHECK-LLVM: %[[shift:[0-9]+]] = lshr i64 %[[bitcast]], 32
; CHECK-LLVM: call void @llvm.dbg.value(metadata i64 %[[shift]], metadata !{{[0-9]+}}, metadata !DIExpression()), !dbg !{{[0-9]+}}
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
