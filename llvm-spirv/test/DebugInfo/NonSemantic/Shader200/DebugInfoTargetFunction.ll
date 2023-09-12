; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -spirv-text %t.bc -o %t.spt --spirv-debug-info-version=nonsemantic-shader-200
; RUN: FileCheck < %t.spt %s -check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -to-binary %t.spt -o %t.spv

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s -check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: ExtInstImport [[#EISId:]] "NonSemantic.Shader.DebugInfo.200"
; CHECK-SPIRV-DAG: String [[#Func:]] "foo_wrapper"
; CHECK-SPIRV-DAG: String [[#TargetFunc:]] "_Z3foov"

; CHECK-SPIRV-DAG: TypeInt [[#TyInt32Id:]] 32 0
; CHECK-SPIRV-DAG: Constant [[#TyInt32Id]] [[#Constant1Id:]] 1{{[[:space:]]}}
; CHECK-SPIRV-DAG: Constant [[#TyInt32Id]] [[#Constant2Id:]] 2{{[[:space:]]}}
; CHECK-SPIRV-DAG: Constant [[#TyInt32Id]] [[#Constant4Id:]] 4{{[[:space:]]}}
; CHECK-SPIRV-DAG: Constant [[#TyInt32Id]] [[#Constant5Id:]] 5{{[[:space:]]}}
; CHECK-SPIRV-DAG: Constant [[#TyInt32Id]] [[#Constant6Id:]] 6{{[[:space:]]}}
; CHECK-SPIRV-DAG: Constant [[#TyInt32Id]] [[#Constant8Id:]] 8{{[[:space:]]}}
; CHECK-SPIRV-DAG: Constant [[#TyInt32Id]] [[#Constant9Id:]] 9{{[[:space:]]}}

; CHECK-SPIRV-DAG: ExtInst [[#]] [[#DebugNone:]] [[#]] DebugInfoNone
; CHECK-SPIRV-DAG: ExtInst [[#]] [[#]] [[#]] DebugFunction [[#Func]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#DebugNone]] [[#TargetFunc]]

; CHECK-SPIRV: [[#EISId]] DebugLine [[#]] [[#Constant4Id]] [[#Constant4Id]] [[#Constant5Id]] [[#Constant6Id]]
; CHECK-SPIRV: [[#EISId]] DebugLine [[#]] [[#Constant5Id]] [[#Constant5Id]] [[#Constant1Id]] [[#Constant2Id]]
; CHECK-SPIRV: [[#EISId]] DebugLine [[#]] [[#Constant8Id]] [[#Constant8Id]] [[#Constant5Id]] [[#Constant6Id]]
; CHECK-SPIRV: [[#EISId]] DebugLine [[#]] [[#Constant9Id]] [[#Constant9Id]] [[#Constant1Id]] [[#Constant2Id]]

; CHECK-LLVM: define spir_func void @_Z11foo_wrapperv() {{.*}} !dbg ![[#DbgSubProg:]] {
; CHECK-LLVM: ![[#Scope_foo_wrapper:]] = distinct !DISubprogram(name: "foo_wrapper", linkageName: "_Z11foo_wrapperv", scope: null, file: ![[#]], line: 3, type: ![[#]], scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: ![[#]], templateParams: ![[#]], targetFuncName: "_Z3foov")
; CHECK-LLVM: !DILocation(line: 4, column: 5, scope: ![[#Scope_foo_wrapper]]
; CHECK-LLVM: !DILocation(line: 5, column: 1, scope: ![[#Scope_foo_wrapper]]
; CHECK-LLVM: ![[#Scope_boo:]] = distinct !DISubprogram(name: "boo", linkageName: "_Z3boov"
; CHECK-LLVM: !DILocation(line: 8, column: 5, scope: ![[#Scope_boo]]
; CHECK-LLVM: !DILocation(line: 9, column: 1, scope: ![[#Scope_boo]]

; ModuleID = 'example.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

define spir_func void @_Z11foo_wrapperv() !dbg !10 {
  call void @_Z3foov(), !dbg !15
  ret void, !dbg !16
}

declare spir_func void @_Z3foov()

define spir_func void @_Z3boov() !dbg !17 {
  call void @_Z11foo_wrapperv(), !dbg !18
  ret void, !dbg !19
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 17.0.0 (https://github.com/llvm/llvm-project.git 88bd2601c013e349fa907b3f878312a94e16e9f6)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/app/example.cpp", directory: "/app")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang version 17.0.0 (https://github.com/llvm/llvm-project.git 88bd2601c013e349fa907b3f878312a94e16e9f6)"}
!10 = distinct !DISubprogram(name: "foo_wrapper", linkageName: "_Z11foo_wrapperv", scope: !11, file: !11, line: 3, type: !12, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14, targetFuncName: "_Z3foov")
!11 = !DIFile(filename: "example.cpp", directory: "/app")
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = !{}
!15 = !DILocation(line: 4, column: 5, scope: !10)
!16 = !DILocation(line: 5, column: 1, scope: !10)
!17 = distinct !DISubprogram(name: "boo", linkageName: "_Z3boov", scope: !11, file: !11, line: 7, type: !12, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !14)
!18 = !DILocation(line: 8, column: 5, scope: !17)
!19 = !DILocation(line: 9, column: 1, scope: !17)
