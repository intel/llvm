; RUN: opt -enable-new-pm=0 -globaloffset %s -S -o - | FileCheck %s
; ModuleID = 'simple_debug.bc'
source_filename = "simple_debug.ll"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda-sycldevice"

; This test checks that debug information on functions and callsites are preserved

declare i32* @llvm.nvvm.implicit.offset()
; CHECK-NOT: llvm.nvvm.implicit.offset

define weak_odr dso_local i64 @_ZTS14other_function() !dbg !11 {
; CHECK: define weak_odr dso_local i64 @_ZTS14other_function(i32* %0) !dbg !11 {
  %1 = tail call i32* @llvm.nvvm.implicit.offset()
  %2 = getelementptr inbounds i32, i32* %1, i64 2
  %3 = load i32, i32* %2, align 4
  %4 = zext i32 %3 to i64
  ret i64 %4
}

; Function Attrs: noinline
define weak_odr dso_local void @_ZTS14example_kernel() !dbg !14 {
; CHECK: define weak_odr dso_local void @_ZTS14example_kernel() !dbg !14 {
entry:
  %0 = call i64 @_ZTS14other_function(), !dbg !15
; CHECK: %3 = call i64 @_ZTS14other_function(i32* %2), !dbg !15
  ret void
}

; CHECK: define weak_odr dso_local void @_ZTS14example_kernel_with_offset([3 x i32]* byval([3 x i32]) %0) !dbg !16 {
; CHECK:   %2 = call i64 @_ZTS14other_function(i32* %1), !dbg !17

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!nvvm.annotations = !{!5, !6, !7, !6, !8, !8, !8, !8, !9, !9, !8}
!nvvmir.version = !{!6}

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
