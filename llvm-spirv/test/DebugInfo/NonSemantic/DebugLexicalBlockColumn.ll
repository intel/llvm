; RUN: llvm-spirv --spirv-debug-info-version=nonsemantic-shader-100 \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info %s -o %t.100.spv
; RUN: llvm-spirv -r %t.100.spv -o %t.100.rev.bc
; RUN: llvm-dis %t.100.rev.bc -o %t.100.rev.ll
; RUN: FileCheck %s --input-file %t.100.rev.ll

; RUN: llvm-spirv --spirv-debug-info-version=nonsemantic-shader-200 \
; RUN:   --spirv-ext=+SPV_KHR_non_semantic_info %s -o %t.200.spv
; RUN: llvm-spirv -r %t.200.spv -o %t.200.rev.bc
; RUN: llvm-dis %t.200.rev.bc -o %t.200.rev.ll
; RUN: FileCheck %s --input-file %t.200.rev.ll

; CHECK-DAG: distinct !DILexicalBlock(scope: ![[#]], file: ![[#]], line: 5, column: 7)
; CHECK-DAG: distinct !DILexicalBlock(scope: ![[#]], file: ![[#]])

target triple = "spir64-unknown-unknown"

define spir_kernel void @test(i32 %a) !dbg !6 {
entry:
  %add = add i32 %a, 1, !dbg !10
  ret void, !dbg !12
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "repro", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "repro.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !7, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = distinct !DILexicalBlock(scope: !6, file: !1, line: 5, column: 7)
!10 = !DILocation(line: 6, column: 3, scope: !9)
!11 = distinct !DILexicalBlock(scope: !6, file: !1)
!12 = !DILocation(line: 9, scope: !11)
