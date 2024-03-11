; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text --spirv-preserve-auxdata -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: Capability
; CHECK-SPIRV-NOT: NonSemanticAuxData
; CHECK-SPIRV: FunctionEnd
target triple = "spir64-unknown-unknown"

define spir_func void @foo() #1 !dbg !4 {
ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang 16.0.0", isOptimized: false, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "foo.c", directory: "./")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, unit: !0, retainedNodes: !5)
!5 = !{!9}
!6 = !DISubroutineType(types: !7)
!7 = !{!8, !8, !8}
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DILabel(scope: !4, name: "top", file: !1, line: 4)
!10 = !DILocation(line: 4, column: 1, scope: !4)
!11 = !DILabel(scope: !4, name: "done", file: !1, line: 7)
!12 = !DILocation(line: 7, column: 1, scope: !4)
