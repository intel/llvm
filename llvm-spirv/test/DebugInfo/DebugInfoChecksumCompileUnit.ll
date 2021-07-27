; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck %s --input-file %t.rev.ll --check-prefix CHECK-LLVM

; ModuleID = 'array-transform.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

; CHECK-LLVM: !DIFile(filename: "array-transform.cpp"
; CHECK-LLVM-SAME: checksumkind: CSK_MD5, checksum: "7768106c1e51aa084de0ffae6fbe50c4"
; CHECK-SPIRV: String [[#ChecksumInfo:]] "//__CSK_MD5:7768106c1e51aa084de0ffae6fbe50c4"
; CHECK-SPIRV: DebugSource
; CHECK-SPIRV-SAME: [[#ChecksumInfo]]

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !2, producer: "spirv", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !3, imports: !3)
!2 = !DIFile(filename: "array-transform.cpp", directory: "D:\\path\\to", checksumkind: CSK_MD5, checksum: "7768106c1e51aa084de0ffae6fbe50c4")
!3 = !{}
