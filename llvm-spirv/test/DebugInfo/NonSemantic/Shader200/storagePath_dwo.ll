; Test checks that dwoId and splitDebugFilename is preserved from LLVM IR to spirv
; and spirv to LLVM IR translation.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-debug-info-version=nonsemantic-shader-200 -spirv-text -o - | FileCheck %s --check-prefix CHECK-SPIRV
; RUN: llvm-spirv %t.bc --spirv-debug-info-version=nonsemantic-shader-200 -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck %s --input-file %t.rev.ll --check-prefix CHECK-LLVM

; CHECK-SPIRV: String [[#stringA_id:]] "11111"
; CHECK-SPIRV: String [[#stringA_sf:]] "debugA_info.dwo"
; CHECK-SPIRV: [[#buildID_A:]] [[#]] DebugBuildIdentifier [[#stringA_id]]
; CHECK-SPIRV: [[#storageID_A:]] [[#]] DebugStoragePath [[#stringA_sf]]

; CHECK-LLVM: !DICompileUnit
; CHECK-LLVM-SAME: splitDebugFilename: "debugA_info.dwo"
; CHECK-LLVM-SAME: dwoId: 11111
; CHECK-LLVM: !DICompileUnit
; CHECK-LLVM-SAME: splitDebugFilename: "debugA_info.dwo"
; CHECK-LLVM-SAME: dwoId: 11111

!llvm.dbg.cu = !{!7, !0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Clang", isOptimized: false, runtimeVersion: 2, splitDebugFilename: "debugA_info.dwo", emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, imports: !2, dwoId: 11111)
!1 = !DIFile(filename: "<stdin>", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!6}
!6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!7 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "Clang", isOptimized: false, runtimeVersion: 2, splitDebugFilename: "debugA_info.dwo", dwoId: 11111, emissionKind: FullDebug, retainedTypes: !5)
