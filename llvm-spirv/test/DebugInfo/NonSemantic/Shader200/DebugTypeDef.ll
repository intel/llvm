; This test covers assertion "Typedef should have a parent scope"

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text --spirv-debug-info-version=nonsemantic-shader-200 -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-debug-info-version=nonsemantic-shader-200
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: String [[#TestStr:]] "test.cpp"
; CHECK-SPIRV: String [[#StdStr:]] "stddef.h"
; CHECK-SPIRV: [[#SourceCpp:]] [[#]] DebugSource [[#TestStr]]
; CHECK-SPIRV: [[#Scope:]] [[#]] DebugCompilationUnit [[#]] [[#]] [[#SourceCpp]]
; CHECK-SPIRV: [[#SourceHeader:]] [[#]] DebugSource [[#StdStr]]
; CHECK-SPIRV: DebugTypedef [[#]] [[#]] [[#SourceHeader]] [[#]] [[#]] [[#Scope]]
; CHECK-SPIRV: DebugEntryPoint [[#]] [[#Scope]]

; CHECK-LLVM: DW_TAG_typedef
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

$_ZTSZ4mainEUlvE_ = comdat any

; Function Attrs: convergent mustprogress norecurse nounwind
define weak_odr dso_local spir_kernel void @_ZTSZ4mainEUlvE_() #0 comdat !dbg !14 {
entry:
  ret void
}

attributes #0 = { convergent mustprogress norecurse nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10, !11, !12, !13}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang based compiler", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3, imports: !2, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariableExpression(var: !5, expr: !DIExpression(DW_OP_constu, 175, DW_OP_stack_value))
!5 = distinct !DIGlobalVariable(name: "N", scope: !0, file: !1, line: 7, type: !6, isLocal: true, isDefinition: true)
!6 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !7)
!7 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !8, line: 62, baseType: !9)
!8 = !DIFile(filename: "stddef.h", directory: "")
!9 = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
!10 = !{i32 7, !"Dwarf Version", i32 4}
!11 = !{i32 2, !"Debug Info Version", i32 3}
!12 = !{i32 1, !"wchar_size", i32 4}
!13 = !{i32 7, !"frame-pointer", i32 2}
!14 = distinct !DISubprogram(name: "_ZTSZ4mainEUlvE_", scope: !1, file: !1, line: 24, type: !15, flags: DIFlagArtificial | DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!15 = !DISubroutineType(cc: DW_CC_nocall, types: !16)
!16 = !{null}
!17 = !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: !18, entity: !19, file: !1, line: 58)
!18 = !DINamespace(name: "std", scope: null)
!19 = !DIDerivedType(tag: DW_TAG_typedef, name: "max_align_t", file: !1, baseType: !20)
!20 = !DICompositeType(tag: DW_TAG_structure_type, file: !8, line: 19, size: 256, flags: DIFlagFwdDecl, elements: !2, identifier: "_ZTS11max_align_t")
