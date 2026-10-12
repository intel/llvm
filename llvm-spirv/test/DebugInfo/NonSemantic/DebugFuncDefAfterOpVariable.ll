; RUN: llvm-spirv %s --spirv-debug-info-version=nonsemantic-shader-100 -spirv-text -o - | FileCheck %s --check-prefix CHECK-SPIRV
; RUN: llvm-spirv %s --spirv-debug-info-version=nonsemantic-shader-200 -spirv-text -o - | FileCheck %s --check-prefix CHECK-SPIRV

; Test that DebugFunctionDefinition is inserted after OpVariable instructions.

; CHECK-SPIRV: Variable [[#]] [[#]] [[#]]
; CHECK-SPIRV-NEXT: Variable [[#]] [[#]] [[#]]
; CHECK-SPIRV-NEXT: ExtInst [[#]] [[#]] [[#]] DebugFunctionDefinition

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

define void @foo() !dbg !4 {
entry:
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  ret void, !dbg !7
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !3, nameTableKind: None)
!1 = !DIFile(filename: "foo.cpp", directory: "/app")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !3)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DILocation(line: 2, column: 1, scope: !4)
