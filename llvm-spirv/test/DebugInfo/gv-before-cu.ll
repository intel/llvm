; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.ll
; RUN: FileCheck < %t.ll %s

; CHECK: [[#GVExpr:]] = !DIGlobalVariableExpression(var: ![[#GV:]], expr: !DIExpression())
; CHECK: ![[#GV]] = distinct !DIGlobalVariable(scope: ![[#CU:]], file: ![[#File:]], line: 1, type: ![[#]], isLocal: true, isDefinition: true)
; CHECK: ![[#CU]] = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_17, file: ![[#File]], producer: "C++ compiler", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: ![[#GVs:]])
; CHECK: ![[#File]] = !DIFile(filename: "test.cpp", directory: "/dev/null")
; CHECK: ![[#GVs]] = !{![[#GVExpr]]}

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@.str = internal unnamed_addr addrspace(2) constant [3 x i8] c"bar", align 1, !dbg !0

define spir_func void @_Z3barBase() !dbg !12 {
entry:
  ret void
}

define spir_kernel void @_Z3fooBase() !dbg !13 {
entry:
  call spir_func void @_Z3barBase(), !dbg !15
  ret void
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!10, !11}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(scope: !2, file: !3, line: 1, type: !5, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "C++ compiler", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4)
!3 = !DIFile(filename: "test.cpp", directory: "/dev/null")
!4 = !{!0}
!5 = !DICompositeType(tag: DW_TAG_array_type, baseType: !6, size: 104, elements: !8)
!6 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !7)
!7 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!8 = !{!9}
!9 = !DISubrange(count: 3, lowerBound: 0)
!10 = !{i32 7, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barBase", scope: null, file: !3, line: 1, type: null, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2)
!13 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooBase", scope: null, file: !3, line: 3, type: null, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2)
!14 = distinct !DILexicalBlock(scope: !13, file: !3, line: 3, column: 1)
!15 = !DILocation(line: 3, column: 1, scope: !14)
