; Check if the translator handles #dbg_declare(ptr null ...) correctly

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefixes=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-debug-info-version=nonsemantic-shader-100
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefixes=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-debug-info-version=nonsemantic-shader-200
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefixes=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: ExtInst [[#]] [[#None:]] [[#]] DebugInfoNone
; CHECK-SPIRV: ExtInst [[#]] [[#LocalVar:]] [[#]] DebugLocalVariable
; CHECK-SPIRV: ExtInst [[#]] [[#]] [[#]] DebugDeclare [[#LocalVar]] [[#None]] [[#]]

; CHECK-LLVM: #dbg_declare(ptr null, ![[#LocalVar:]], !DIExpression(DW_OP_constu, 4, DW_OP_swap, DW_OP_xderef), ![[#Loc:]])
; CHECK-LLVM-DAG: ![[#LocalVar]] = !DILocalVariable(name: "bar"
; CHECK-LLVM-DAG: ![[#Loc]] = !DILocation(line: 23

; ModuleID = 'test.bc'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64"

define spir_kernel void @__omp_offloading_811_29c0007__Z4main_l28() {
newFuncRoot:
  %0 = alloca i32, i32 0, align 4, !dbg !4
  store i32 0, ptr null, align 4
  %four.ascast.fpriv = alloca i32, align 4, !dbg !4
    #dbg_declare(ptr null, !8, !DIExpression(DW_OP_constu, 4, DW_OP_swap, DW_OP_xderef), !10)
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, imports: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/path/to")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DILocation(line: 28, column: 1, scope: !5)
!5 = distinct !DILexicalBlock(scope: !6, file: !1, line: 28, column: 1)
!6 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 28, type: !7, scopeLine: 28, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0)
!7 = !DISubroutineType(types: !2)
!8 = !DILocalVariable(name: "bar", scope: !5, file: !1, line: 23, type: !9)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DILocation(line: 23, column: 7, scope: !5)

