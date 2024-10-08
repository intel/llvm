; This test checks that DW_OP_LLVM_arg operation goes through round trip translation correctly.
; DW_OP_LLVM_arg is mapped on 165 in SPIR-V

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-allow-extra-diexpressions
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefixes=CHECK-SPIRV-OCL
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM-OCL

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-debug-info-version=nonsemantic-shader-200
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefixes=CHECK-SPIRV-200
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM-200

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-allow-extra-diexpressions --experimental-debuginfo-iterators=1
; RUN: llvm-spirv %t.spv -to-text -o %t.spt --experimental-debuginfo-iterators=1
; RUN: FileCheck < %t.spt %s --check-prefixes=CHECK-SPIRV-OCL
; RUN: llvm-spirv -r --experimental-debuginfo-iterators=1 %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM-OCL

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-debug-info-version=nonsemantic-shader-200 --experimental-debuginfo-iterators=1
; RUN: llvm-spirv %t.spv -to-text -o %t.spt --experimental-debuginfo-iterators=1
; RUN: FileCheck < %t.spt %s --check-prefixes=CHECK-SPIRV-200
; RUN: llvm-spirv -r --experimental-debuginfo-iterators=1 %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM-200

; CHECK-SPIRV-200-DAG: TypeInt [[#INT32:]] 32 0
; CHECK-SPIRV-200-DAG: Constant [[#INT32:]] [[#CONST1:]] 1 {{$}}
; CHECK-SPIRV-200-DAG: Constant [[#INT32]] [[#CONST0:]] 0
; CHECK-SPIRV-200-DAG: Constant [[#INT32]] [[#CONST165:]] 165
; CHECK-SPIRV-200-NOT: Undef
; CHECK-SPIRV-200: [[#DEBUG_LOC_VAR:]] [[#]] DebugLocalVariable
; CHECK-SPIRV-200: [[#EXPR_1_ARG_0:]] [[#]] DebugOperation [[#CONST165]] [[#CONST0]]
; CHECK-SPIRV-200: [[#EXPR_1:]] [[#]] DebugExpression [[#EXPR_1_ARG_0]]
; CHECK-SPIRV-200: [[#EXPR_2_ARG_0:]] [[#]] DebugOperation [[#CONST165]] [[#CONST0]]
; CHECK-SPIRV-200: [[#EXPR_2_ARG_1:]] [[#]] DebugOperation [[#CONST165]] [[#CONST1]]
; CHECK-SPIRV-200: [[#EXPR_2_ARG_2:]] [[#]] DebugOperation [[#CONST1:]]
; CHECK-SPIRV-200: [[#EXPR_2:]] [[#]] DebugExpression [[#EXPR_2_ARG_0]] [[#EXPR_2_ARG_1]] [[#EXPR_2_ARG_2]]
; CHECK-SPIRV-200: Variable [[#]] [[#VAL:]]
; CHECK-SPIRV-200: DebugValue [[#DEBUG_LOC_VAR]] [[#VAL]] [[#EXPR_1]]
; CHECK-SPIRV-200: DebugValue [[#DEBUG_LOC_VAR]] [[#VAL]] [[#EXPR_2]]

; CHECK-SPIRV-OCL: Undef [[#]] [[#UNDEF:]]
; CHECK-SPIRV-OCL: [[#DEBUG_LOC_VAR:]] [[#]] DebugLocalVariable
; CHECK-SPIRV-OCL: [[#EXPR_ARG_0:]] [[#]] DebugOperation 165 0
; CHECK-SPIRV-OCL: [[#EXPRESSION:]] [[#]] DebugExpression [[#EXPR_ARG_0]]
; CHECK-SPIRV-OCL: [[#EXPR_EMPTY:]] [[#]] DebugExpression{{ *$}}
; CHECK-SPIRV-OCL: Variable [[#]] [[#VAL:]]
; CHECK-SPIRV-OCL: DebugValue [[#DEBUG_LOC_VAR]] [[#VAL]] [[#EXPRESSION]]
; CHECK-SPIRV-OCL: DebugValue [[#DEBUG_LOC_VAR]] [[#UNDEF]] [[#EXPR_EMPTY]]

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare void @llvm.dbg.value(metadata, metadata, metadata) nounwind readnone speculatable

define void @DbgIntrinsics() sanitize_memtag {
entry:
  %x = alloca i32, align 4
; CHECK-LLVM-OCL: #dbg_value(!DIArgList(ptr %x), ![[#]], !DIExpression(DW_OP_LLVM_arg, 0), ![[#]])
; CHECK-LLVM-200: #dbg_value(!DIArgList(ptr %x), ![[#]], !DIExpression(DW_OP_LLVM_arg, 0), ![[#]])
  call void @llvm.dbg.value(metadata !DIArgList(ptr %x), metadata !6, metadata !DIExpression(DW_OP_LLVM_arg, 0)), !dbg !10
; CHECK-LLVM-OCL: #dbg_value(ptr undef, ![[#]], !DIExpression(), ![[#]])
; CHECK-LLVM-200: #dbg_value(!DIArgList(ptr %x, ptr %x), ![[#]], !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus)
  call void @llvm.dbg.value(metadata !DIArgList(ptr %x, ptr %x), metadata !6, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus)), !dbg !10
  store i32 42, ptr %x, align 4
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 9.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "stack-tagging.cc", directory: "/tmp")
!2 = !{}
!3 = distinct !DISubprogram(name: "DbgIntrinsics", linkageName: "DbgIntrinsics", scope: !1, file: !1, line: 3, type: !4, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!4 = !DISubroutineType(types: !5)
!5 = !{null}
!6 = !DILocalVariable(name: "x", scope: !3, file: !1, line: 4, type: !7)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{i32 2, !"Dwarf Version", i32 4}
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = !DILocation(line: 1, column: 2, scope: !3)
