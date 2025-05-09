; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --check-globals all --include-generated-funcs --version 5
; RUN: opt -bugpoint-enable-legacy-pm -globaloffset %s -S -o - | FileCheck %s

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

; This test checks that debug information on functions and callsites are preserved

declare ptr addrspace(5) @llvm.amdgcn.implicit.offset()

define i64 @_ZTS14other_function() !dbg !11 {
  %1 = tail call ptr addrspace(5) @llvm.amdgcn.implicit.offset()
  %2 = getelementptr inbounds i32, ptr addrspace(5) %1, i64 2
  %3 = load i32, ptr addrspace(5) %2, align 4
  %4 = zext i32 %3 to i64
  ret i64 %4
}

define amdgpu_kernel void @_ZTS14example_kernel() !dbg !14 {
entry:
  %0 = call i64 @_ZTS14other_function(), !dbg !15
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 0.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "global-offset-debug.cpp", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"sycl-device", i32 1}
!11 = distinct !DISubprogram(name: "other_function", scope: !1, file: !1, line: 3, type: !12, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = distinct !DISubprogram(name: "example_kernel", scope: !1, file: !1, line: 10, type: !12, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!15 = !DILocation(line: 1, column: 2, scope: !14)
; CHECK-LABEL: define i64 @_ZTS14other_function(
; CHECK-SAME: ) !dbg [[DBG6:![0-9]+]] {
; CHECK-NEXT:    [[TMP1:%.*]] = zext i32 0 to i64
; CHECK-NEXT:    ret i64 [[TMP1]]
;
;
; CHECK-LABEL: define i64 @_ZTS14other_function_with_offset(
; CHECK-SAME: ptr addrspace(5) [[TMP0:%.*]]) !dbg [[DBG9:![0-9]+]] {
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i32, ptr addrspace(5) [[TMP0]], i64 2
; CHECK-NEXT:    [[TMP3:%.*]] = load i32, ptr addrspace(5) [[TMP2]], align 4
; CHECK-NEXT:    [[TMP4:%.*]] = zext i32 [[TMP3]] to i64
; CHECK-NEXT:    ret i64 [[TMP4]]
;
;
; CHECK-LABEL: define amdgpu_kernel void @_ZTS14example_kernel(
; CHECK-SAME: ) !dbg [[DBG10:![0-9]+]] {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[TMP0:%.*]] = call i64 @_ZTS14other_function(), !dbg [[DBG11:![0-9]+]]
; CHECK-NEXT:    ret void
;
;
; CHECK-LABEL: define amdgpu_kernel void @_ZTS14example_kernel_with_offset(
; CHECK-SAME: ptr byref([3 x i32]) [[TMP0:%.*]]) !dbg [[DBG12:![0-9]+]] {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[TMP1:%.*]] = alloca [3 x i32], align 4, addrspace(5), !dbg [[DBG13:![0-9]+]]
; CHECK-NEXT:    [[TMP2:%.*]] = addrspacecast ptr [[TMP0]] to ptr addrspace(4), !dbg [[DBG13]]
; CHECK-NEXT:    call void @llvm.memcpy.p5.p4.i64(ptr addrspace(5) align 4 [[TMP1]], ptr addrspace(4) align 1 [[TMP2]], i64 12, i1 false), !dbg [[DBG13]]
; CHECK-NEXT:    [[TMP3:%.*]] = call i64 @_ZTS14other_function_with_offset(ptr addrspace(5) [[TMP1]]), !dbg [[DBG13]]
; CHECK-NEXT:    ret void
;
;.
; CHECK: attributes #[[ATTR0:[0-9]+]] = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
;.
; CHECK: [[META0:![0-9]+]] = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: [[META1:![0-9]+]], producer: "{{.*}}clang version {{.*}}", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: [[META2:![0-9]+]], nameTableKind: None)
; CHECK: [[META1]] = !DIFile(filename: "global-offset-debug.cpp", directory: {{.*}})
; CHECK: [[META2]] = !{}
; CHECK: [[META3:![0-9]+]] = !{i32 2, !"Dwarf Version", i32 4}
; CHECK: [[META4:![0-9]+]] = !{i32 2, !"Debug Info Version", i32 3}
; CHECK: [[META5:![0-9]+]] = !{i32 1, !"sycl-device", i32 1}
; CHECK: [[DBG6]] = distinct !DISubprogram(name: "other_function", scope: [[META1]], file: [[META1]], line: 3, type: [[META7:![0-9]+]], scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: [[META0]], retainedNodes: [[META2]])
; CHECK: [[META7]] = !DISubroutineType(types: [[META8:![0-9]+]])
; CHECK: [[META8]] = !{null}
; CHECK: [[DBG9]] = distinct !DISubprogram(name: "other_function", scope: [[META1]], file: [[META1]], line: 3, type: [[META7]], scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: [[META0]], retainedNodes: [[META2]])
; CHECK: [[DBG10]] = distinct !DISubprogram(name: "example_kernel", scope: [[META1]], file: [[META1]], line: 10, type: [[META7]], scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: [[META0]], retainedNodes: [[META2]])
; CHECK: [[DBG11]] = !DILocation(line: 1, column: 2, scope: [[DBG10]])
; CHECK: [[DBG12]] = distinct !DISubprogram(name: "example_kernel", scope: [[META1]], file: [[META1]], line: 10, type: [[META7]], scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: [[META0]], retainedNodes: [[META2]])
; CHECK: [[DBG13]] = !DILocation(line: 1, column: 2, scope: [[DBG12]])
;.
