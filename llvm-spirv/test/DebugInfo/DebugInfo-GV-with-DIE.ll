;; Ensure that DIExpressions are preserved in DIGlobalVariableExpressions
;; if nonsemantic debug info is enabled.
;; This utilizes SPIRV DebugGlobalVariable's Variable field to hold the
;; DIExpression.

; RUN: llvm-as %s -o %t.bc

; RUN: llvm-spirv -o %t.100.spt %t.bc --spirv-debug-info-version=nonsemantic-shader-100 -spirv-text
; RUN: FileCheck %s --input-file %t.100.spt --check-prefix CHECK-SPIRV
; RUN: llvm-spirv -o %t.100.spv %t.bc --spirv-debug-info-version=nonsemantic-shader-100
; RUN: llvm-spirv -r -o %t.100.rev.bc %t.100.spv
; RUN: llvm-dis %t.100.rev.bc -o %t.100.rev.ll
; RUN: FileCheck %s --input-file %t.100.rev.ll --check-prefix CHECK-LLVM

; RUN: llvm-spirv -o %t.200.spt %t.bc --spirv-debug-info-version=nonsemantic-shader-200 -spirv-text
; RUN: FileCheck %s --input-file %t.200.spt --check-prefix CHECK-SPIRV
; RUN: llvm-spirv -o %t.200.spv %t.bc --spirv-debug-info-version=nonsemantic-shader-200
; RUN: llvm-spirv -r -o %t.200.rev.bc %t.200.spv
; RUN: llvm-dis %t.200.rev.bc -o %t.200.rev.ll
; RUN: FileCheck %s --input-file %t.200.rev.ll --check-prefix CHECK-LLVM

; CHECK-SPIRV: [[EXPRESSION:[0-9]+]] [[#]] DebugExpression [[#]] [[#]]
; CHECK-SPIRV: [[#]] [[#]] DebugGlobalVariable [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[EXPRESSION]] [[#]] {{$}}

; CHECK-LLVM: ![[#]] = !DIGlobalVariableExpression(var: ![[#GV:]], expr: !DIExpression(DW_OP_constu, 1, DW_OP_stack_value))
; CHECK-LLVM: ![[#GV]] = distinct !DIGlobalVariable(name: "true", scope: ![[#]], file: ![[#]], line: 3777, type: ![[#]], isLocal: true, isDefinition: true)

;; Ensure SPIR-V DebugGlobalVariable's Variable field does not hold a DIExpression if nonsemantic debug info is not enabled

; RUN: llvm-spirv -o %t.spt %t.bc -spirv-text
; RUN: FileCheck %s --input-file %t.spt --check-prefix CHECK-NONE-SPIRV

; CHECK-NONE-SPIRV: [[DEBUG_INFO_NONE:[0-9]+]] [[#]] DebugInfoNone
; CHECK-NONE-SPIRV: [[#]] [[#]] DebugGlobalVariable [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[DEBUG_INFO_NONE]] [[#]] {{$}}

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 7, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang", emissionKind: FullDebug, globals: !4)
!3 = !DIFile(filename: "test.cpp", directory: "/path/to")
!4 = !{!5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression(DW_OP_constu, 1, DW_OP_stack_value))
!6 = distinct !DIGlobalVariable(name: "true", scope: !2, file: !3, line: 3777, type: !7, isLocal: true, isDefinition: true)
!7 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !8)
!8 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
