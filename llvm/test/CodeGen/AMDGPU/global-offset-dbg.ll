; RUN: opt -bugpoint-enable-legacy-pm -globaloffset %s -S -o - | FileCheck %s
; ModuleID = 'simple_debug.bc'
source_filename = "global-offset-dbg.ll"

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

; This test checks that debug information on functions and callsites are preserved

declare i32 addrspace(5)* @llvm.amdgcn.implicit.offset()
; CHECK-NOT: llvm.amdgcn.implicit.offset

define weak_odr dso_local i64 @_ZTS14other_function() !dbg !11 {
; CHECK: define weak_odr dso_local i64 @_ZTS14other_function(i32 addrspace(5)* %0) !dbg !11 {
  %1 = tail call i32 addrspace(5)* @llvm.amdgcn.implicit.offset()
  %2 = getelementptr inbounds i32, i32 addrspace(5)* %1, i64 2
  %3 = load i32, i32 addrspace(5)* %2, align 4
  %4 = zext i32 %3 to i64
  ret i64 %4
}

; Function Attrs: noinline
define weak_odr dso_local void @_ZTS14example_kernel() !dbg !14 {
; CHECK: define weak_odr dso_local void @_ZTS14example_kernel() !dbg !14 {
entry:
  %0 = call i64 @_ZTS14other_function(), !dbg !15
; CHECK: %3 = call i64 @_ZTS14other_function(i32 addrspace(5)* %2), !dbg !15
  ret void
}

; CHECK: define weak_odr dso_local void @_ZTS14example_kernel_with_offset([3 x i32]* byref([3 x i32]) %0) !dbg !16 {
; CHECK:  %1 = alloca [3 x i32], align 4, addrspace(5), !dbg !17
; CHECK:  %2 = bitcast [3 x i32] addrspace(5)* %1 to i32 addrspace(5)*, !dbg !17
; CHECK:  call void @llvm.memcpy.p5i8.p4i8.i64(i8 addrspace(5)* align 4 %4, i8 addrspace(4)* align 1 %3, i64 12, i1 false), !dbg !17
; CHECK:  %5 = call i64 @_ZTS14other_function(i32 addrspace(5)* %2), !dbg !17

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!amdgcn.annotations = !{!5, !6, !7, !6, !8, !8, !8, !8, !9, !9, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 0.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "global-offset-debug.cpp", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !{void ()* @_ZTS14example_kernel, !"kernel", i32 1}
!6 = !{i32 1, i32 4}
!7 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!8 = !{null, !"align", i32 16}
!9 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!11 = distinct !DISubprogram(name: "other_function", scope: !1, file: !1, line: 3, type: !12, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = distinct !DISubprogram(name: "example_kernel", scope: !1, file: !1, line: 10, type: !12, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!15 = !DILocation(line: 1, column: 2, scope: !14)
; CHECK: !16 = distinct !DISubprogram(name: "example_kernel", scope: !1, file: !1, line: 10, type: !12, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
; CHECK: !17 = !DILocation(line: 1, column: 2, scope: !16)
