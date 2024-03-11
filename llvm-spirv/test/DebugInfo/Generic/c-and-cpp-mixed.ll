;; This test checks that two DICompileUnits resulted in a link of C and C++
;; object files are being translated correctly

; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv --to-text %t.spv -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.ll
; RUN: FileCheck < %t.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: String [[#Foo:]] "foo"
; CHECK-SPIRV: String [[#Main:]] "main"
; CHECK-SPIRV: ExtInst [[#]] [[#CU1:]] [[#]] DebugCompilationUnit
; CHECK-SPIRV: ExtInst [[#]] [[#CU2:]] [[#]] DebugCompilationUnit
; CHECK-SPIRV: ExtInst [[#]] [[#Func1:]] [[#]] DebugFunction [[#Foo]] [[#]] [[#]] [[#]] [[#]] [[#CU1]]
; CHECK-SPIRV: ExtInst [[#]] [[#Func2:]] [[#]] DebugFunction [[#Main]] [[#]] [[#]] [[#]] [[#]] [[#CU2]]
; CHECK-SPIRV: ExtInst [[#]] [[#]] [[#]] DebugLexicalBlock [[#]] [[#]] [[#]] [[#Func1]]
; CHECK-SPIRV: ExtInst [[#]] [[#]] [[#]] DebugLexicalBlock [[#]] [[#]] [[#]] [[#Func2]]

; CHECK-LLVM: define spir_func void @foo() #0 !dbg ![[#Func1:]] {
; CHECK-LLVM: entry:
; CHECK-LLVM:   %puts = call spir_func i32 @puts(ptr addrspace(1) nocapture @str) #0, !dbg ![[#Puts1Loc:]]
; CHECK-LLVM:   ret void, !dbg ![[#Ret1:]]
; CHECK-LLVM: }

; CHECK-LLVM: define spir_func i32 @main(i32 %argc, ptr nocapture %argv) #0 !dbg ![[#Func2:]] {
; CHECK-LLVM: entry:
; CHECK-LLVM:   call void @llvm.dbg.value(metadata i32 %argc, metadata ![[#Fun2Param1:]], metadata !DIExpression()), !dbg ![[#Fun2Param1Loc:]]
; CHECK-LLVM:   call void @llvm.dbg.value(metadata ptr %argv, metadata ![[#Fun2Param2:]], metadata !DIExpression(DW_OP_deref, DW_OP_deref)), !dbg ![[#Fun2Param2Loc:]]
; CHECK-LLVM:   %0 = bitcast ptr addrspace(1) @str1 to ptr addrspace(1), !dbg ![[#Puts2Loc:]]
; CHECK-LLVM:   %puts = call spir_func i32 @puts(ptr addrspace(1) nocapture %0) #0, !dbg ![[#Puts2Loc]]
; CHECK-LLVM:   call spir_func void @foo() #0, !dbg ![[#CallFoo:]]
; CHECK-LLVM:   ret i32 0, !dbg ![[#Ret2:]]
; CHECK-LLVM: }

; CHECK-LLVM: !llvm.dbg.cu = !{![[#CU1:]], ![[#CU2:]]}
; CHECK-LLVM: ![[#CU1]] = distinct !DICompileUnit(language: DW_LANG_OpenCL, file: ![[#File:]], producer: "clang version 17", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
; CHECK-LLVM: ![[#File]] = !DIFile(filename: "foo.c", directory: "/tmp")
; CHECK-LLVM: ![[#CU2]] = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_17, file: ![[#File]], producer: "clang version 17", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
; CHECK-LLVM: ![[#Empty:]] = !{}
; CHECK-LLVM: ![[#Func1]] = distinct !DISubprogram(name: "foo", scope: null, file: ![[#File]], line: 5, type: ![[#Func1T:]], scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: ![[#CU1]], templateParams: ![[#Empty]])
; CHECK-LLVM: ![[#Func1T]] = !DISubroutineType(types: ![[#Func1TP:]])
; CHECK-LLVM: ![[#Func1TP]] = !{null}
; CHECK-LLVM: ![[#Puts1Loc]] = !DILocation(line: 6, column: 3, scope: ![[#Puts1Scope:]])
; CHECK-LLVM: ![[#Puts1Scope]] = distinct !DILexicalBlock(scope: ![[#Func1]], file: ![[#File]], line: 5, column: 16)
; CHECK-LLVM: ![[#Ret1]] = !DILocation(line: 7, column: 1, scope: ![[#Puts1Scope]])
; CHECK-LLVM: ![[#Func2]] = distinct !DISubprogram(name: "main", scope: null, file: ![[#File]], line: 11, type: ![[#Func2T:]], scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: ![[#CU2]], templateParams: ![[#Empty]], retainedNodes: ![[#Fun2Params:]])
; CHECK-LLVM: ![[#Func2T]] = !DISubroutineType(types: ![[#Func2TP:]])
; CHECK-LLVM: ![[#Func2TP]] = !{![[#Func2TP1:]],
; CHECK-LLVM-SAME: ![[#Func2TP1]], ![[#Func2TP2:]]
; CHECK-LLVM: ![[#Func2TP1]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
; CHECK-LLVM: ![[#Func2TP2]] = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
; CHECK-LLVM: ![[#Fun2Params]] = !{![[#Fun2Param1]], ![[#Fun2Param2]]}
; CHECK-LLVM: ![[#Fun2Param1]] = !DILocalVariable(name: "argc", arg: 1, scope: ![[#Func2]], file: ![[#File]], line: 11, type: ![[#Func2TP1]])
; CHECK-LLVM: ![[#Fun2Param2]] = !DILocalVariable(name: "argv", arg: 2, scope: ![[#Func2]], file: ![[#File]], line: 11, type: ![[#Func2TP2]])
; CHECK-LLVM: ![[#Fun2Param1Loc:]] = !DILocation(line: 11, column: 14, scope: ![[#Func2]])
; CHECK-LLVM: ![[#Fun2Param2Loc:]] = !DILocation(line: 11, column: 26, scope: ![[#Func2]])
; CHECK-LLVM: ![[#Puts2Loc]] = !DILocation(line: 12, column: 3, scope: ![[#Puts2Scope:]]
; CHECK-LLVM: ![[#Puts2Scope]] = distinct !DILexicalBlock(scope: ![[#Func2]], file: ![[#File]], line: 11, column: 34)
; CHECK-LLVM: ![[#CallFoo]] = !DILocation(line: 13, column: 3, scope: ![[#Puts2Scope]])
; CHECK-LLVM: ![[#Ret2]] = !DILocation(line: 14, column: 3, scope: ![[#Puts2Scope]])


target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; ModuleID = 'test.bc'

@str = private unnamed_addr addrspace(1) constant [4 x i8] c"FOO\00"
@str1 = private unnamed_addr addrspace(1) constant [6 x i8] c"Main!\00"

define void @foo() nounwind !dbg !5 {
entry:
  %puts = tail call i32 @puts(ptr addrspace(1) @str), !dbg !23
  ret void, !dbg !25
}

declare i32 @puts(ptr addrspace(1) nocapture) nounwind

define i32 @main(i32 %argc, ptr nocapture %argv) nounwind !dbg !12 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %argc, metadata !21, metadata !DIExpression()), !dbg !26
  ; Avoid talking about the pointer size in debug info because that's target dependent
  tail call void @llvm.dbg.value(metadata ptr %argv, metadata !22, metadata !DIExpression(DW_OP_deref, DW_OP_deref)), !dbg !27
  %puts = tail call i32 @puts(ptr addrspace(1) @str1), !dbg !28
  tail call void @foo() nounwind, !dbg !30
  ret i32 0, !dbg !31
}

declare void @llvm.dbg.value(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0, !9}
!llvm.module.flags = !{!33}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 17", isOptimized: true, emissionKind: FullDebug, file: !32, enums: !1, retainedTypes: !1, globals: !1, imports: !1)
!1 = !{}
!5 = distinct !DISubprogram(name: "foo", line: 5, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 5, file: !32, scope: !6, type: !7, retainedNodes: !1)
!6 = !DIFile(filename: "foo.c", directory: "/tmp")
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_17, producer: "clang version 17", isOptimized: true, emissionKind: FullDebug, file: !32, enums: !1, retainedTypes: !1, globals: !1, imports: !1)
!12 = distinct !DISubprogram(name: "main", line: 11, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !9, scopeLine: 11, file: !32, scope: !6, type: !13, retainedNodes: !19)
!13 = !DISubroutineType(types: !14)
!14 = !{!15, !15, !18}
!15 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!18 = !DIBasicType(tag: DW_TAG_base_type, name: "char", size: 8, align: 8, encoding: DW_ATE_signed_char)
!19 = !{!21, !22}
!21 = !DILocalVariable(name: "argc", line: 11, arg: 1, scope: !12, file: !6, type: !15)
!22 = !DILocalVariable(name: "argv", line: 11, arg: 2, scope: !12, file: !6, type: !18)
!23 = !DILocation(line: 6, column: 3, scope: !24)
!24 = distinct !DILexicalBlock(line: 5, column: 16, file: !32, scope: !5)
!25 = !DILocation(line: 7, column: 1, scope: !24)
!26 = !DILocation(line: 11, column: 14, scope: !12)
!27 = !DILocation(line: 11, column: 26, scope: !12)
!28 = !DILocation(line: 12, column: 3, scope: !29)
!29 = distinct !DILexicalBlock(line: 11, column: 34, file: !32, scope: !12)
!30 = !DILocation(line: 13, column: 3, scope: !29)
!31 = !DILocation(line: 14, column: 3, scope: !29)
!32 = !DIFile(filename: "foo.c", directory: "/tmp")
!33 = !{i32 1, !"Debug Info Version", i32 3}
