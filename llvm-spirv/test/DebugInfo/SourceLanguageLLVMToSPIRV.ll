; Test checks that DW_LANG_C99 and DW_LANG_OpenCL are mapped to OpenCL C
; in the SourceLanguage enum, and also that DW_LANG_C_plus_plus and
; DW_LANG_C_plus_plus_14 are mapped to C++ for OpenCL.

; LLVM does not yet define DW_LANG_C_plus_plus_17. A test case that
; checks DW_LANG_C_plus_plus_17 is mapped to C++ for OpenCL should be
; added once this is defined.

; RUN: sed -e 's/INPUT_LANGUAGE/DW_LANG_C99/' %s | llvm-as - -o %t.bc
; RUN: llvm-spirv -spirv-text %t.bc -o - | FileCheck %s --check-prefix=CHECK-OPENCLC

; RUN: sed -e 's/INPUT_LANGUAGE/DW_LANG_OpenCL/' %s | llvm-as - -o %t.bc
; RUN: llvm-spirv -spirv-text %t.bc -o - | FileCheck %s --check-prefix=CHECK-OPENCLC

; RUN: sed -e 's/INPUT_LANGUAGE/DW_LANG_C_plus_plus/' %s | llvm-as - -o %t.bc
; RUN: llvm-spirv -spirv-text %t.bc -o - | FileCheck %s --check-prefix=CHECK-CPP4OPENCL

; RUN: sed -e 's/INPUT_LANGUAGE/DW_LANG_C_plus_plus_14/' %s | llvm-as - -o %t.bc
; RUN: llvm-spirv -spirv-text %t.bc -o - | FileCheck %s --check-prefix=CHECK-CPP4OPENCL

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

; CHECK-OPENCLC: DebugCompileUnit {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 3
; CHECK-CPP4OPENCL: DebugCompileUnit {{[0-9]+}} {{[0-9]+}} {{[0-9]+}} 6
