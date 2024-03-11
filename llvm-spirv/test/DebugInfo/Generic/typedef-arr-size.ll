; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Check that the size of vectors is translated

source_filename = "test/DebugInfo/Generic/typedef-arr-size.ll"

@x = dso_local addrspace(1) global <16 x i32> zeroinitializer, align 16, !dbg !0

!llvm.dbg.cu = !{!8}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "x", scope: null, file: !2, line: 2, type: !5, isLocal: false, isDefinition: true)
!2 = !DIFile(filename: "typedef-arr-size.cpp", directory: "/tmp/dbginfo")
!3 = !{!4}
!4 = !DISubrange(count: 16)
; CHECK: DICompositeType(tag: DW_TAG_array_type, {{.*}}, size: 512, flags: DIFlagVector,
!5 = !DICompositeType(tag: DW_TAG_array_type, baseType: !6, size: 512, flags: DIFlagVector, elements: !3)
!6 = !DIDerivedType(tag: DW_TAG_typedef, name: "data_t", file: !2, line: 42, baseType: !7)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang version 3.5.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !9, retainedTypes: !9, globals: !10, imports: !9)
!9 = !{}
!10 = !{!0}
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 1, !"Debug Info Version", i32 3}
!13 = !{!"clang version 3.5.0 "}

