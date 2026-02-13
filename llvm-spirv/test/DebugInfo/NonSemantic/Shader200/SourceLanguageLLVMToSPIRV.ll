; Test checks that DW_LANG_C99, DW_LANG_OpenCL, and all DW_LANG_C_plus_plus_X are mapped to
; appropriate SourceLanguages in SPIRV when the extended debug info is enabled.

; RUN: sed -e 's/INPUT_LANGUAGE/DW_LANG_C99/' %s | llvm-as - -o %t.bc
; RUN: llvm-spirv --spirv-debug-info-version=nonsemantic-shader-200 -spirv-text %t.bc -o - | FileCheck %s --check-prefix=CHECK-C99

; RUN: sed -e 's/INPUT_LANGUAGE/DW_LANG_OpenCL/' %s | llvm-as - -o %t.bc
; RUN: llvm-spirv --spirv-debug-info-version=nonsemantic-shader-200 -spirv-text %t.bc -o - | FileCheck %s --check-prefix=CHECK-OPENCLC

; RUN: sed -e 's/INPUT_LANGUAGE/DW_LANG_C_plus_plus/' %s | llvm-as - -o %t.bc
; RUN: llvm-spirv --spirv-debug-info-version=nonsemantic-shader-200 -spirv-text %t.bc -o - | FileCheck %s --check-prefix=CHECK-CPP

; RUN: sed -e 's/INPUT_LANGUAGE/DW_LANG_C_plus_plus_14/' %s | llvm-as - -o %t.bc
; RUN: llvm-spirv --spirv-debug-info-version=nonsemantic-shader-200 -spirv-text %t.bc -o - | FileCheck %s --check-prefix=CHECK-CPP14

; RUN: sed -e 's/INPUT_LANGUAGE/DW_LANG_C_plus_plus_17/' %s | llvm-as - -o %t.bc
; RUN: llvm-spirv --spirv-debug-info-version=nonsemantic-shader-200 -spirv-text %t.bc -o - | FileCheck %s --check-prefix=CHECK-CPP17

; CHECK-C99: Constant [[#]] [[#Constant:]] 211
; CHECK-C99: DebugCompilationUnit [[#]] [[#]] [[#]] [[#Constant]]
; CHECK-OPENCLC: Constant [[#]] [[#Constant:]] 3
; CHECK-OPENCLC: DebugCompilationUnit [[#]] [[#]] [[#]] [[#Constant]]
; CHECK-CPP: Constant [[#]] [[#Constant:]] 214
; CHECK-CPP: DebugCompilationUnit [[#]] [[#]] [[#]] [[#Constant]]
; CHECK-CPP14: Constant [[#]] [[#Constant:]] 217
; CHECK-CPP14: DebugCompilationUnit [[#]] [[#]] [[#]] [[#Constant]]
; CHECK-CPP17: Constant [[#]] [[#Constant:]] 218
; CHECK-CPP17: DebugCompilationUnit [[#]] [[#]] [[#]] [[#Constant]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

define dso_local spir_kernel void @func() local_unnamed_addr !dbg !7 !kernel_arg_addr_space !2 !kernel_arg_access_qual !2 !kernel_arg_type !2 !kernel_arg_base_type !2 !kernel_arg_type_qual !2 {
entry:
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: INPUT_LANGUAGE, file: !1)
!1 = !DIFile(filename: "test.cl", directory: "/tmp", checksumkind: CSK_MD5, checksum: "18aa9ce738eaafc7b7b7181c19092815")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 2, i32 0}
!7 = distinct !DISubprogram(name: "func", scope: !8, file: !8, line: 1, scopeLine: 2, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!8 = !DIFile(filename: "test.cl", directory: "/tmp", checksumkind: CSK_MD5, checksum: "18aa9ce738eaafc7b7b7181c19092815")
