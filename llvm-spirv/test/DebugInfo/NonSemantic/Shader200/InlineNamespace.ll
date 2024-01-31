; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -spirv-text %t.bc -o %t.spt --spirv-debug-info-version=nonsemantic-shader-200
; RUN: FileCheck < %t.spt %s -check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -to-binary %t.spt -o %t.spv

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s -check-prefix=CHECK-LLVM

; Source:
; -- inline-namespace.cxx -----------------------------------------------------
;      1  inline namespace Outer {
;      2    namespace Inner {
;      3      int global;
;      4    }
;      5  }
;      6
;      7  void foo() {
;      8    Inner::global++;
;      9  }
; -- inline-namespace.cxx -----------------------------------------------------

; CHECK-SPIRV: String [[#StrOuter:]] "Outer"
; CHECK-SPIRV: String [[#StrInner:]] "Inner"
; CHECK-SPIRV: TypeBool [[#TypeBool:]]
; CHECK-SPIRV: ConstantTrue [[#TypeBool]] [[#ConstTrue:]]
; CHECK-SPIRV: ConstantFalse [[#TypeBool]] [[#ConstFalse:]]
; CHECK-SPIRV: ExtInst [[#]] [[#]] [[#]] DebugLexicalBlock [[#]] [[#]] [[#]] [[#]] [[#StrOuter]] [[#ConstTrue]]
; CHECK-SPIRV: ExtInst [[#]] [[#]] [[#]] DebugLexicalBlock [[#]] [[#]] [[#]] [[#]] [[#StrInner]] [[#ConstFalse]]

; CHECK-LLVM: !DINamespace(name: "Inner", scope: ![[#OuterSpace:]])
; CHECK-LLVM: ![[#OuterSpace]] = !DINamespace(name: "Outer", scope: null, exportSymbols: true)

; ModuleID = 'inline-namespace.cxx'
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

@_ZN5Outer5Inner6globalE = dso_local addrspace(1) global i32 0, align 4, !dbg !0

; Function Attrs: noinline nounwind optnone uwtable mustprogress
define dso_local void @_Z3foov() #0 !dbg !11 {
entry:
  %0 = load i32, ptr addrspace(1) @_ZN5Outer5Inner6globalE, align 4, !dbg !14
  %inc = add nsw i32 %0, 1, !dbg !14
  store i32 %inc, ptr addrspace(1) @_ZN5Outer5Inner6globalE, align 4, !dbg !14
  ret void, !dbg !15
}

attributes #0 = { noinline nounwind optnone uwtable mustprogress }

!llvm.dbg.cu = !{!6}
!llvm.module.flags = !{!9, !10}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "global", linkageName: "_ZN5Outer5Inner6globalE", scope: !2, file: !4, line: 3, type: !5, isLocal: false, isDefinition: true)
!2 = !DINamespace(name: "Inner", scope: !3)
!3 = !DINamespace(name: "Outer", scope: null, exportSymbols: true)
!4 = !DIFile(filename: "inlined-namespace.cxx", directory: "/path/to")
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !4, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !7, globals: !8, splitDebugInlining: false, nameTableKind: None)
!7 = !{}
!8 = !{!0}
!9 = !{i32 7, !"Dwarf Version", i32 4}
!10 = !{i32 2, !"Debug Info Version", i32 3}
!11 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !4, file: !4, line: 7, type: !12, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !7)
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = !DILocation(line: 8, column: 16, scope: !11)
!15 = !DILocation(line: 9, column: 1, scope: !11)
