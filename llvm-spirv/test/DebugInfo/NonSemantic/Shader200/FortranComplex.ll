;; Tests for Fortran's complex type encoding in debug info
;; Compiled from the following Fortran source
;;
;; program complex_numbers
;;   implicit none
;;   complex :: a, b, c
;;
;;   a = (1.0, 2.0)
;;   b = (2.0, -1.0)
;;
;;   c = a + b
;;
;; end program complex_numbers

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text --spirv-debug-info-version=nonsemantic-shader-200 -o - | FileCheck %s --check-prefix=CHECK-SPIRV-200
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-debug-info-version=nonsemantic-shader-200
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM-200

; RUN: llvm-spirv %t.bc -spirv-text --spirv-debug-info-version=nonsemantic-shader-100 -o - | FileCheck %s --check-prefix=CHECK-SPIRV-100
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-debug-info-version=nonsemantic-shader-100
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM-100

; CHECK-SPIRV-200-DAG: ExtInstImport [[#Import:]] "NonSemantic.Shader.DebugInfo.200
; CHECK-SPIRV-200-DAG: String [[#Name:]] "COMPLEX*8"
; CHECK-SPIRV-200-DAG: Constant [[#]] [[#Size:]] 64
; CHECK-SPIRV-200-DAG: Constant [[#]] [[#Encoding:]] 8
; CHECK-SPIRV-200-DAG: ExtInst [[#]] [[#Type:]] [[#Import]] DebugTypeBasic [[#Name]] [[#Size]] [[#Encoding]]
; CHECK-SPIRV-200-DAG: ExtInst [[#]] [[#]] [[#Import]] DebugLocalVariable [[#]] [[#Type]]
; CHECK-SPIRV-200-DAG: ExtInst [[#]] [[#]] [[#Import]] DebugLocalVariable [[#]] [[#Type]]
; CHECK-SPIRV-200-DAG: ExtInst [[#]] [[#]] [[#Import]] DebugLocalVariable [[#]] [[#Type]]

; CHECK-LLVM-200-DAG: ![[#]] = !DILocalVariable(name: "a", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[#Type:]])
; CHECK-LLVM-200-DAG: ![[#Type]] = !DIBasicType(name: "COMPLEX*8", size: 64, encoding: DW_ATE_complex_float)
; CHECK-LLVM-200-DAG: ![[#]] = !DILocalVariable(name: "b", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[#Type]])
; CHECK-LLVM-200-DAG: ![[#]] = !DILocalVariable(name: "c", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[#Type]])

; CHECK-SPIRV-100-DAG: ExtInstImport [[#Import:]] "NonSemantic.Shader.DebugInfo.100
; CHECK-SPIRV-100-DAG: String [[#Name:]] "COMPLEX*8"
; CHECK-SPIRV-100-DAG: Constant [[#]] [[#Size:]] 64
; CHECK-SPIRV-100-DAG: Constant [[#]] [[#Encoding:]] 0
; CHECK-SPIRV-100-DAG: ExtInst [[#]] [[#Type:]] [[#Import]] DebugTypeBasic [[#Name]] [[#Size]] [[#Encoding]]
; CHECK-SPIRV-100-DAG: ExtInst [[#]] [[#]] [[#Import]] DebugLocalVariable [[#]] [[#Type]]
; CHECK-SPIRV-100-DAG: ExtInst [[#]] [[#]] [[#Import]] DebugLocalVariable [[#]] [[#Type]]
; CHECK-SPIRV-100-DAG: ExtInst [[#]] [[#]] [[#Import]] DebugLocalVariable [[#]] [[#Type]]

; CHECK-LLVM-100-DAG: ![[#]] = !DILocalVariable(name: "a", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[#Type:]])
; CHECK-LLVM-100-DAG: ![[#Type]] = !DIBasicType(tag: DW_TAG_unspecified_type, name: "COMPLEX*8")
; CHECK-LLVM-100-DAG: ![[#]] = !DILocalVariable(name: "b", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[#Type]])
; CHECK-LLVM-100-DAG: ![[#]] = !DILocalVariable(name: "c", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[#Type]])

; ModuleID = 'test.f90'
source_filename = "test.f90"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

@0 = internal unnamed_addr addrspace(1) constant i32 65536, align 4
@1 = internal unnamed_addr addrspace(1) constant i32 2, align 4

; Function Attrs: nounwind uwtable
define void @MAIN__() local_unnamed_addr !dbg !4 !llfort.type_idx !12 {
alloca_0:
  %func_result = tail call i32 @for_set_fpe_(ptr addrspace(1) nonnull @0), !dbg !13, !llfort.type_idx !14
  %func_result2 = tail call i32 @for_set_reentrancy(ptr addrspace(1) nonnull @1), !dbg !13, !llfort.type_idx !14
  call void @llvm.dbg.value(metadata float 1.000000e+00, metadata !11, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32)), !dbg !15
  call void @llvm.dbg.value(metadata float 2.000000e+00, metadata !11, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32)), !dbg !15
  call void @llvm.dbg.value(metadata float 2.000000e+00, metadata !10, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32)), !dbg !15
  call void @llvm.dbg.value(metadata float -1.000000e+00, metadata !10, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32)), !dbg !15
  call void @llvm.dbg.value(metadata float poison, metadata !8, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 32)), !dbg !15
  call void @llvm.dbg.value(metadata float poison, metadata !8, metadata !DIExpression(DW_OP_LLVM_fragment, 32, 32)), !dbg !15
  ret void, !dbg !16
}

declare !llfort.intrin_id !17 !llfort.type_idx !18 i32 @for_set_fpe_(ptr addrspace(1) nocapture readonly) local_unnamed_addr

; Function Attrs: nofree
declare !llfort.intrin_id !19 !llfort.type_idx !20 i32 @for_set_reentrancy(ptr addrspace(1) nocapture readonly) local_unnamed_addr

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}
!omp_offload.info = !{}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 2, !"Dwarf Version", i32 4}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !3, producer: "Intel(R) Fortran", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "test.f90", directory: "complex")
!4 = distinct !DISubprogram(name: "complex_numbers", linkageName: "MAIN__", scope: !3, file: !3, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2, retainedNodes: !7)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{!8, !10, !11}
!8 = !DILocalVariable(name: "c", scope: !4, file: !3, line: 3, type: !9)
!9 = !DIBasicType(name: "COMPLEX*8", size: 64, encoding: DW_ATE_complex_float)
!10 = !DILocalVariable(name: "b", scope: !4, file: !3, line: 3, type: !9)
!11 = !DILocalVariable(name: "a", scope: !4, file: !3, line: 3, type: !9)
!12 = !{i64 23}
!13 = !DILocation(line: 1, column: 9, scope: !4)
!14 = !{i64 2}
!15 = !DILocation(line: 0, scope: !4)
!16 = !DILocation(line: 9, column: 1, scope: !4)
!17 = !{i32 97}
!18 = !{i64 27}
!19 = !{i32 98}
!20 = !{i64 29}
