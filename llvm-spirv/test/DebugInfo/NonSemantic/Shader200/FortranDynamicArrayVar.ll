;; DebugInfo/dwarfdump-dataLocationVar.ll from llvm.org is used as base for this test
;; The test checks, that Fortran dynamic arrays are being correctly represented
;; by SPIR-V debug information
;; Unlike 'static' arrays dynamic can have following parameters of
;; DICompositeType metadata with DW_TAG_array_type tag:
;; Data Location, Associated, Allocated and Rank which can be represented
;; by either DIExpression or DIVariable (both local and global).
;; This test if for variable representation.
;; FortranDynamicArrayVar.ll is for expression representation.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text --spirv-debug-info-version=nonsemantic-shader-200 -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-debug-info-version=nonsemantic-shader-200
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

;; Major SPIR-V checks are done in FortranDynamicArrayExpr.ll
; CHECK-SPIRV: ExtInst [[#]] [[#DbgLVarId:]] [[#]] DebugLocalVariable [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] {{$}}
; CHECK-SPIRV: ExtInst [[#]] [[#]] [[#]] DebugTypeArrayDynamic [[#]] [[#DbgLVarId]] [[#]] [[#]] [[#]] [[#]]

; CHECK-LLVM: %[[#Ptr:]] = alloca ptr
; CHECK-LLVM: %[[#Array:]] = alloca [16 x i64]
; CHECK-LLVM: #dbg_declare(ptr %[[#Array]], ![[#DbgLVarArray:]]
; CHECK-LLVM: #dbg_declare(ptr %[[#Ptr]], ![[#DbgLVarPtr:]]

; CHECK-LLVM: ![[#DbgLVarPtr:]] = !DILocalVariable(scope: ![[#]], file: ![[#]], type: ![[#DbgPtrT:]], flags: DIFlagArtificial)
; CHECK-LLVM: ![[#DbgPtrT:]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[#DbgBasicT:]], size: 64)
; CHECK-LLVM: ![[#DbgBasicT]] = !DIBasicType(name: "integer", size: 32, encoding: DW_ATE_signed)
; CHECK-LLVM: ![[#DbgLVarArray]] = !DILocalVariable(name: "arr", scope: ![[#]], file: ![[#]], type: ![[#DbgArrayT:]])
; CHECK-LLVM: ![[#DbgArrayT]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[#DbgBasicT]], size: 608, elements: ![[#Elements:]], dataLocation: ![[#DbgLVarPtr]])
; CHECK-LLVM: ![[#Elements]] = !{![[#SubRange:]]}
; CHECK-LLVM: ![[#SubRange]] = !DISubrange(count: 19, lowerBound: 2)

; ModuleID = 'fortsubrange.ll'
source_filename = "fortsubrange.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_func void @foo() !dbg !5 {
L.entry:
  %0 = alloca ptr, align 8
  %1 = alloca [16 x i64], align 8
  call void @llvm.dbg.declare(metadata ptr %1, metadata !8, metadata !DIExpression()), !dbg !16
  call void @llvm.dbg.declare(metadata ptr %0, metadata !14, metadata !DIExpression()), !dbg !16
  ret void, !dbg !17
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata)

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !3, producer: " F95 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "fortsubrange.f90", directory: "/dir")
!4 = !{}
!5 = distinct !DISubprogram(name: "main", scope: !2, file: !3, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "arr", scope: !9, file: !3, type: !10)
!9 = !DILexicalBlock(scope: !5, file: !3, line: 1, column: 1)
!10 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 32, align: 32, elements: !12, dataLocation: !14)
!11 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!12 = !{!13}
!13 = !DISubrange(count: 19, lowerBound: 2)
!14 = distinct !DILocalVariable(scope: !9, file: !3, type: !15, flags: DIFlagArtificial)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !11, size: 32, align: 32)
!16 = !DILocation(line: 0, scope: !9)
!17 = !DILocation(line: 6, column: 1, scope: !9)
