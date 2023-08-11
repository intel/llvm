; ModuleID = '/Volumes/Data/apple-internal/llvm/tools/clang/test/Modules/debug-info-moduleimport.m'
; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv --spirv-debug-info-version=nonsemantic-shader-200 %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.ll

; RUN: llc -mtriple=x86_64-apple-macosx %t.ll -accel-tables=Dwarf -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-info %t | FileCheck %s
; RUN: llvm-dwarfdump -verify %t

; RUN: llvm-spirv --spirv-debug-info-version=nonsemantic-shader-200 %t.bc -spirv-text -o %t.spt
; RUN: FileCheck %s --input-file %t.spt --check-prefix CHECK-SPIRV

; CHECK: DW_TAG_compile_unit
; CHECK-NOT: DW_TAG
; CHECK:   DW_TAG_module
; CHECK-NEXT: DW_AT_name {{.*}}"DebugModule"
; CHECK-NEXT: DW_AT_LLVM_config_macros {{.*}}"-DMODULES=0"
; CHECK-NEXT: DW_AT_LLVM_include_path {{.*}}"/llvm/tools/clang/test/Modules/Inputs"
; CHECK-NEXT: DW_AT_LLVM_apinotes {{.*}}"m.apinotes"

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; CHECK-SPIRV-DAG: ExtInstImport [[#EISId:]] "NonSemantic.Shader.DebugInfo.200"
; CHECK-SPIRV: String [[#FileName:]] "/llvm/tools/clang/test/Modules/<stdin>"
; CHECK-SPIRV: String [[#EmptyStr:]] ""
; CHECK-SPIRV: String [[#Name:]] "DebugModule"
; CHECK-SPIRV: String [[#Defines:]] "-DMODULES=0"
; CHECK-SPIRV: String [[#IncludePath:]] "/llvm/tools/clang/test/Modules/Inputs"
; CHECK-SPIRV: String [[#ApiNotes:]] "m.apinotes"
; CHECK-SPIRV: TypeInt [[#TypeInt32:]] 32 0
; CHECK-SPIRV-DAG: Constant [[#TypeInt32]] [[#Constant0:]] 0
; CHECK-SPIRV-DAG: Constant [[#TypeInt32]] [[#Version:]] 65536
; CHECK-SPIRV-DAG: Constant [[#TypeInt32]] [[#DWARF:]] 4

; CHECK-SPIRV: ExtInst [[#]] [[#Source:]] [[#]] DebugSource [[#FileName]]
; CHECK-SPIRV: ExtInst [[#]] [[#Parent:]] [[#]] DebugCompilationUnit [[#Version]] [[#DWARF]]
; CHECK-SPIRV: ExtInst [[#]] [[#SourceEmpty:]] [[#]] DebugSource [[#EmptyStr]]
; CHECK-SPIRV: ExtInst [[#]] [[#Module:]] [[#]] DebugModule [[#Name]] [[#SourceEmpty]] [[#Constant0]] [[#Parent]] [[#Defines]] [[#IncludePath]] [[#ApiNotes]] [[#Constant0]]
; CHECK-SPIRV: ExtInst [[#]] [[#]] [[#]] DebugImportedEntity [[#]] [[#]] [[#Source]] [[#Module]]

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_ObjC, file: !1, producer: "LLVM version 3.7.0", isOptimized: false, runtimeVersion: 2, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, imports: !3,  sysroot: "/")
!1 = !DIFile(filename: "/llvm/tools/clang/test/Modules/<stdin>", directory: "/")
!2 = !{}
!3 = !{!4}
!4 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !0, entity: !5, file: !1, line: 5)
!5 = !DIModule(scope: null, name: "DebugModule", configMacros: "-DMODULES=0", includePath: "/llvm/tools/clang/test/Modules/Inputs", apinotes: "m.apinotes")
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{!"LLVM version 3.7.0"}
