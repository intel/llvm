; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv -spirv-debug-info-version=nonsemantic-shader-100
; RUN: llvm-spirv %t.spv --to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: llvm-spirv %t.bc -o %t.spv -spirv-debug-info-version=nonsemantic-shader-200
; RUN: llvm-spirv %t.spv --to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; Generated from:
;
; struct A {
;   static int fully_specified;
;   static int smem[];
; };
;
; int A::fully_specified;
; int A::smem[] = { 0, 1, 2, 3 };

; CHECK-SPIRV: ExtInst [[#]] [[#Member1:]] [[#]] DebugTypeMember [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] {{$}}
; CHECK-SPIRV: ExtInst [[#]] [[#Member2:]] [[#]] DebugTypeMember [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] {{$}} 
; CHECK-SPIRV: ExtInst [[#]] [[#]] [[#]] DebugTypeComposite [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#Member1]] [[#Member2]]
; CHECK-SPIRV: ExtInst [[#]] [[#]] [[#]] DebugGlobalVariable [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#Member1]]
; CHECK-SPIRV: ExtInst [[#]] [[#]] [[#]] DebugGlobalVariable [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#Member2]]

; CHECK-LLVM: ![[#GVExpr1:]] = !DIGlobalVariableExpression(var: ![[#GV1:]], expr: !DIExpression())
; CHECK-LLVM: ![[#GV1]] = distinct !DIGlobalVariable(name: "fully_specified", linkageName: "_ZN1A15fully_specifiedE", scope: ![[#CU:]], file: ![[#File:]], line: 7, type: ![[#GVTy1:]], isLocal: false, isDefinition: true, declaration: ![[#Decl1:]])
; CHECK-LLVM: ![[#CU]] = distinct !DICompileUnit(language: DW_LANG_C_plus_plus{{.*}}, file: ![[#File:]], {{.*}}, globals: ![[#GVs:]]
; CHECK-LLVM: ![[#File]] = !DIFile(filename: "static_member_array.cpp", directory: "/Volumes/Data/radar/28706946")
; CHECK-LLVM: ![[#GVs]] = !{![[#GVExpr1]], ![[#GVExpr2:]]}
; CHECK-LLVM: ![[#GVExpr2]] = !DIGlobalVariableExpression(var: ![[#GV2:]], expr: !DIExpression())
; CHECK-LLVM: ![[#GV2]] = distinct !DIGlobalVariable(name: "smem", linkageName: "_ZN1A4smemE", scope: ![[#CU]], file: ![[#File]], line: 8, type: ![[#GVTy2:]], isLocal: false, isDefinition: true, declaration: ![[#Decl2:]])
; CHECK-LLVM: ![[#GVTy2]] = !DICompositeType(tag: DW_TAG_array_type, baseType: !8, size: 128, elements: ![[#Elements1:]])
; CHECK-LLVM: ![[#GVTy1]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
; CHECK-LLVM: ![[#Elements1]] = !{![[#Subrange:]]}
; CHECK-LLVM: ![[#Subrange]] = !DISubrange(count: 4
; CHECK-LLVM: ![[#MemTy1:]] = !DIDerivedType(tag: DW_TAG_member, name: "smem", scope: ![[#StructTy:]], file: ![[#File]], line: 4, baseType: ![[#ArrTy:]], flags: DIFlagPublic | DIFlagStaticMember)
; CHECK-LLVM: ![[#StructTy]] = !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: ![[#File]], line: 1, size: 8, elements: ![[#Elements2:]], identifier: "_ZTS1A")
; CHECK-LLVM: ![[#Elements2]] = !{![[#MemTy2:]], ![[#MemTy1]]}
; CHECK-LLVM: ![[#MemTy2:]] = !DIDerivedType(tag: DW_TAG_member, name: "fully_specified", scope: ![[#StructTy]], file: ![[#File]], line: 3, baseType: ![[#GVTy1]], flags: DIFlagPublic | DIFlagStaticMember)
; CHECK-LLVM: ![[#ArrTy]] = !DICompositeType(tag: DW_TAG_array_type, baseType: !8, elements: ![[#]])


source_filename = "static_member_array.cpp"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

@_ZN1A15fully_specifiedE = addrspace(1) global i32 0, align 4, !dbg !0
@_ZN1A4smemE = addrspace(1) global [4 x i32] [i32 0, i32 1, i32 2, i32 3], align 16, !dbg !6

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!19, !20, !21}
!llvm.ident = !{!22}

!0 = distinct !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = !DIGlobalVariable(name: "fully_specified", linkageName: "_ZN1A15fully_specifiedE", scope: !2, file: !3, line: 7, type: !9, isLocal: false, isDefinition: true, declaration: !15)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 4.0.0 (trunk 286129) (llvm/trunk 286128)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "static_member_array.cpp", directory: "/Volumes/Data/radar/28706946")
!4 = !{}
!5 = !{!0, !6}
!6 = distinct !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = !DIGlobalVariable(name: "smem", linkageName: "_ZN1A4smemE", scope: !2, file: !3, line: 8, type: !8, isLocal: false, isDefinition: true, declaration: !12)
!8 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 128, elements: !10)
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{!11}
!11 = !DISubrange(count: 4)
!12 = !DIDerivedType(tag: DW_TAG_member, name: "smem", scope: !13, file: !3, line: 4, baseType: !16, flags: DIFlagStaticMember)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "A", file: !3, line: 1, size: 8, elements: !14, identifier: "_ZTS1A")
!14 = !{!15, !12}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "fully_specified", scope: !13, file: !3, line: 3, baseType: !9, flags: DIFlagStaticMember)
!16 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, elements: !17)
!17 = !{!18}
!18 = !DISubrange(count: -1)
!19 = !{i32 2, !"Dwarf Version", i32 4}
!20 = !{i32 2, !"Debug Info Version", i32 3}
!21 = !{i32 1, !"PIC Level", i32 2}
!22 = !{!"clang version 4.0.0 (trunk 286129) (llvm/trunk 286128)"}

