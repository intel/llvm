; RUN: llvm-as %s -o %t.bc
; Translation shouldn't crash:
; RUN: llvm-spirv %t.bc -spirv-text
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc

; RUN: llvm-spirv -spirv-ext=+SPV_INTEL_debug_module %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc

source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

!llvm.module.flags = !{!1, !2, !3, !4}
!llvm.dbg.cu = !{!5}

!0 = !{}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"PIC Level", i32 2}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 2, !"Dwarf Version", i32 4}
!5 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !6, isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !0, globals: !0, imports: !7, splitDebugInlining: false, nameTableKind: None)
!6 = !DIFile(filename: "declare_target_subroutine.F90", directory: "/test")
!7 = !{!8}
!8 = !DIImportedEntity(tag: DW_TAG_imported_module, scope: !9, entity: !12, file: !6, line: 24)
!9 = distinct !DISubprogram(name: "declare_target_subroutine", linkageName: "MAIN__", scope: !6, file: !6, line: 23, type: !10, scopeLine: 23, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !5, retainedNodes: !0)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !DIModule(scope: !9, name: "iso_fortran_env", isDecl: true)
