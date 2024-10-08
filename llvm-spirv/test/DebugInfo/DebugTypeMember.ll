; Exercise Value field in SPIRV TypeMember.
; It is used to preserve DIDerivedType's extraData argument in
; LLVM IR when a DIFlagStaticMember is present.

; Original *.cpp source
; 
; namespace {
; struct anon_static_decl_struct {
;   static const int anon_static_decl_var = 117 + 234;
; };
; }
; int ref() {
;   return anon_static_decl_struct::anon_static_decl_var;
; }

; RUN: llvm-as %s -o %t.bc

; RUN: llvm-spirv -o %t.spt %t.bc -spirv-text
; RUN: FileCheck %s --input-file %t.spt --check-prefix CHECK-SPIRV
; RUN: llvm-spirv -o %t.spv %t.bc
; RUN: llvm-spirv -r -o %t.rev.bc %t.spv
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck %s --input-file %t.rev.ll --check-prefix CHECK-LLVM

; RUN: llvm-spirv -o %t.100.spt %t.bc --spirv-debug-info-version=nonsemantic-shader-100 -spirv-text
; RUN: FileCheck %s --input-file %t.100.spt --check-prefix CHECK-SPIRV-NONSEMANTIC
; RUN: llvm-spirv -o %t.100.spv %t.bc --spirv-debug-info-version=nonsemantic-shader-100
; RUN: llvm-spirv -r -o %t.100.rev.bc %t.100.spv
; RUN: llvm-dis %t.100.rev.bc -o %t.100.rev.ll
; RUN: FileCheck %s --input-file %t.100.rev.ll --check-prefix CHECK-LLVM

; RUN: llvm-spirv -o %t.200.spt %t.bc --spirv-debug-info-version=nonsemantic-shader-200 -spirv-text
; RUN: FileCheck %s --input-file %t.200.spt --check-prefix CHECK-SPIRV-NONSEMANTIC
; RUN: llvm-spirv -o %t.200.spv %t.bc --spirv-debug-info-version=nonsemantic-shader-200
; RUN: llvm-spirv -r -o %t.200.rev.bc %t.200.spv
; RUN: llvm-dis %t.200.rev.bc -o %t.200.rev.ll
; RUN: FileCheck %s --input-file %t.200.rev.ll --check-prefix CHECK-LLVM

; CHECK-SPIRV: Constant [[#]] [[#VALUE:]] 351
; CHECK-SPIRV: DebugTypeMember [[#NAME:]] [[#TYPE:]] [[#SOURCE:]] [[#LINE:]] [[#COLUMN:]] [[#PARENT:]] [[#OFFSET:]] [[#SIZE:]] [[#FLAGS:]] [[#VALUE:]] {{$}}

; CHECK-SPIRV-NONSEMANTIC: Constant [[#]] [[#VALUE:]] 351
; CHECK-SPIRV-NONSEMANTIC: DebugTypeMember [[#NAME:]] [[#TYPE:]] [[#SOURCE:]] [[#LINE:]] [[#COLUMN:]] [[#OFFSET:]] [[#SIZE:]] [[#FLAGS:]] [[#VALUE:]] {{$}}

; CHECK-LLVM: !DIDerivedType(tag: DW_TAG_member, name: "anon_static_decl_var", scope: ![[#]], file: ![[#]], line: 5, baseType: ![[#]], flags: {{.*}}DIFlagStaticMember{{.*}}, extraData: i32 351)

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "spirv-unknown-unknown"

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef i32 @_Z3refv() !dbg !18 {
entry:
  ret i32 351, !dbg !21
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13, !14, !15, !16}
!llvm.ident = !{!17}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang based Intel(R) oneAPI DPC++/C++ Compiler 2024.1.0 (2024.x.0.YYYYMMDD)", isOptimized: false, flags: " --driver-mode=g++ --intel -g -emit-llvm test.cpp -c -fveclib=SVML -faltmathlib=SVML -fheinous-gnu-extensions", runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !2, globals: !9, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/testdir")
!2 = !{!3}
!3 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "anon_static_decl_struct", scope: !4, file: !1, line: 4, size: 8, flags: DIFlagTypePassByValue, elements: !5)
!4 = !DINamespace(scope: null)
!5 = !{!6}
!6 = !DIDerivedType(tag: DW_TAG_member, name: "anon_static_decl_var", scope: !3, file: !1, line: 5, baseType: !7, flags: DIFlagStaticMember, extraData: i32 351)
!7 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !8)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !{!10}
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression(DW_OP_constu, 351, DW_OP_stack_value))
!11 = distinct !DIGlobalVariable(name: "anon_static_decl_var", scope: !0, file: !1, line: 5, type: !7, isLocal: true, isDefinition: true, declaration: !6)
!12 = !{i32 7, !"Dwarf Version", i32 4}
!13 = !{i32 2, !"Debug Info Version", i32 3}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{i32 7, !"uwtable", i32 2}
!16 = !{i32 7, !"frame-pointer", i32 2}
!17 = !{!"Intel(R) oneAPI DPC++/C++ Compiler 2024.1.0 (2024.x.0.YYYYMMDD)"}
!18 = distinct !DISubprogram(name: "ref", linkageName: "_Z3refv", scope: !1, file: !1, line: 11, type: !19, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!19 = !DISubroutineType(types: !20)
!20 = !{!8}
!21 = !DILocation(line: 12, column: 3, scope: !18)
