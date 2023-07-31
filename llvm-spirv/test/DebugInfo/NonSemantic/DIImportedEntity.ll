; ModuleID = '/Volumes/Data/apple-internal/llvm/tools/clang/test/Modules/debug-info-moduleimport.m'
; RUN: llvm-as < %s -o %t.bc

; RUN: llvm-spirv --spirv-debug-info-version=nonsemantic-shader-100 %t.bc -spirv-text -o %t.spt
; RUN: FileCheck %s --input-file %t.spt --check-prefix CHECK-SPIRV
; RUN: llvm-spirv --spirv-debug-info-version=nonsemantic-shader-100 %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.ll
; RUN: FileCheck %s --input-file %t.ll --check-prefix CHECK-LLVM

; RUN: llvm-spirv --spirv-debug-info-version=nonsemantic-shader-200 %t.bc -spirv-text -o %t.spt
; RUN: FileCheck %s --input-file %t.spt --check-prefix CHECK-SPIRV
; RUN: llvm-spirv --spirv-debug-info-version=nonsemantic-shader-200 %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.ll
; RUN: FileCheck %s --input-file %t.ll --check-prefix CHECK-LLVM

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV-DAG: ExtInstImport [[#EISId:]] "NonSemantic.Shader.DebugInfo
; CHECK-SPIRV-DAG: String [[#Name:]] ""
; CHECK-SPIRV-DAG: TypeInt [[#Int:]] 32 0
; CHECK-SPIRV-DAG: Constant [[#Int]] [[#One:]] 1 
; CHECK-SPIRV-DAG: Constant [[#Int]] [[#Zero:]] 0
; CHECK-SPIRV-DAG: Constant [[#Int]] [[#Five:]] 5
; CHECK-SPIRV: ExtInst [[#]] [[#Source:]] [[#]] DebugSource
; CHECK-SPIRV: ExtInst [[#]] [[#CU:]] [[#]] DebugCompilationUnit
; CHECK-SPIRV: ExtInst [[#]] [[#Typedef:]] [[#]] DebugTypedef
; CHECK-SPIRV: ExtInst [[#]] [[#]] [[#EISId]] DebugImportedEntity [[#Name]] [[#One]] [[#Source]] [[#Typedef]] [[#Five]] [[#Zero]] [[#CU]] {{$}}

; CHECK-LLVM: ![[#CU:]] = distinct !DICompileUnit(language: DW_LANG_OpenCL, file: ![[#File:]],{{.*}}isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, imports: ![[#Import:]])
; CHECK-LLVM: ![[#File]] = !DIFile(filename: "<stdin>", directory: "/llvm/tools/clang/test/Modules")
; CHECK-LLVM: ![[#Import]] = !{![[#Entity:]]}
; CHECK-LLVM: ![[#Entity]] = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: ![[#CU]], entity: ![[#Typedef:]], file: ![[#File]], line: 5)
; CHECK-LLVM: ![[#Typedef]] = !DIDerivedType(tag: DW_TAG_typedef, name: "max_align_t", file: ![[#File]], baseType: ![[#]])
; CHECK-LLVM: !DICompositeType(tag: DW_TAG_structure_type, file: ![[#File]], line: 5, size: 256, flags: DIFlagFwdDecl, elements: ![[#]], identifier: "_ZTS11max_align_t")

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_OpenCL, file: !1, producer: "LLVM version 3.7.0", isOptimized: false, runtimeVersion: 2, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, imports: !3,  sysroot: "/")
!1 = !DIFile(filename: "/llvm/tools/clang/test/Modules/<stdin>", directory: "/")
!2 = !{}
!3 = !{!4}
!4 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !9, file: !1, line: 5)
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{!"LLVM version 3.7.0"}
!9 = !DIDerivedType(tag: DW_TAG_typedef, name: "max_align_t", file: !1, baseType: !10)
!10 = !DICompositeType(tag: DW_TAG_structure_type, file: !1, line: 5, size: 256, flags: DIFlagFwdDecl, elements: !2, identifier: "_ZTS11max_align_t")
