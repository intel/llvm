;; Ensure that DIExpressions are preserved in DIGlobalVariableExpressions
;; when a Static Member Declaration is also needed.
;; This utilizes SPIRV DebugGlobalVariable's Variable field to hold the
;; DIExpression.

;; Declaration generated from:
;;
;; struct A {
;;   static int fully_specified;
;; };
;;  
;; int A::fully_specified;

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

; CHECK-SPIRV-DAG: [[TYPE_MEMBER:[0-9]+]] [[#]] DebugTypeMember [[#]] [[#]] [[#]] [[#]]
; CHECK-SPIRV-DAG: [[EXPRESSION:[0-9]+]] [[#]] DebugExpression [[#]] [[#]]
; CHECK-SPIRV: [[#]] [[#]] DebugGlobalVariable [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[EXPRESSION]] [[#]] [[TYPE_MEMBER]] {{$}}

; CHECK-LLVM: ![[#]] = !DIGlobalVariableExpression(var: ![[#GV:]], expr: !DIExpression(DW_OP_constu, 1, DW_OP_stack_value))
; CHECK-LLVM: ![[#GV]] = distinct !DIGlobalVariable(name: "true", scope: ![[#]], file: ![[#]], line: 3777, type: ![[#]], isLocal: true, isDefinition: true, declaration: ![[#DECLARATION:]])
; CHECK-LLVM: ![[#DECLARATION]] = !DIDerivedType(tag: DW_TAG_member, name: "fully_specified", scope: ![[#SCOPE:]], file: ![[#]], line: 2, baseType: ![[#BASETYPE:]], flags: DIFlagPublic | DIFlagStaticMember)
; CHECK-LLVM: ![[#SCOPE]] = {{.*}}!DICompositeType(tag: DW_TAG_structure_type, name: "A", file: ![[#]], line: 1, size: 8, flags: DIFlagTypePassByValue, elements: ![[#ELEMENTS:]], identifier: "_ZTS1A")
; CHECK-LLVM: ![[#ELEMENTS]] = !{![[#DECLARATION]]}
; CHECK-LLVM: ![[#BASETYPE]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 7, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang", emissionKind: FullDebug, globals: !4)
!3 = !DIFile(filename: "test.cpp", directory: "/path/to")
!4 = !{!5}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression(DW_OP_constu, 1, DW_OP_stack_value))
!6 = distinct !DIGlobalVariable(name: "true", scope: !2, file: !3, line: 3777, type: !7, isLocal: true, isDefinition: true, declaration: !10)
!7 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !8)
!8 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DIDerivedType(tag: DW_TAG_member, name: "fully_specified", scope: !11, file: !3, line: 2, baseType: !9, flags: DIFlagStaticMember)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !3, line: 1, size: 8, flags: DIFlagTypePassByValue, elements: !12, identifier: "_ZTS1A")
!12 = !{!10}
