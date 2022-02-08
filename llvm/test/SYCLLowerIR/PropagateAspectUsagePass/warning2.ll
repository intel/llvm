; RUN: opt --PropagateAspectUsage < %s > %t.ll 2>&1
; RUN: FileCheck %s < %t.ll --check-prefix CHECK-FIRST
; RUN: FileCheck %s < %t.ll --check-prefix CHECK-SECOND
;
; Test checks call chains in traces in FullDebug mode.
; The first check checks a handling of @llvm.dbg.declare intrinsic.
; The second check checks a handling of dbg node attached to instruction.

%MyStruct = type { i32 }

declare void @llvm.dbg.declare(metadata, metadata, metadata)


; CHECK-FIRST: warning: function 'kernel' uses aspect '2' not listed in 'sycl::requires()'
; CHECK-FIRST-NEXT: use is from this call chain:
; CHECK-FIRST-NEXT: kernel() warning2.cpp:5:3
; CHECK-FIRST-NEXT: _Z5func1v() warning2.cpp:10:4
; CHECK-FIRST-NEXT: _Z5func2v() warning2.cpp:15:3


; CHECK-SECOND: warning: function 'kernel' uses aspect '6' not listed in 'sycl::requires()'
; CHECK-SECOND-NEXT: use is from this call chain:
; CHECK-SECOND-NEXT: kernel() warning2.cpp:6:3
; CHECK-SECOND-NEXT: _Z5func3v() warning2.cpp:20:3

define dso_local spir_kernel void @kernel() !intel_declared_aspects !20 !dbg !14 {
  call void @_Z5func1v(), !dbg !18
  call void @_Z5func3v(), !dbg !19
  ret void
}

define spir_func void @_Z5func1v() !dbg !11 {
  call void @_Z5func2v(), !dbg !12
  ret void
}

define spir_func void @_Z5func2v() !dbg !7 {
  %tmp = alloca %MyStruct
  call void @llvm.dbg.declare(metadata %MyStruct* %tmp, metadata !22, metadata !DIExpression()), !dbg !24
  ret void
}

@d = global double 0.000000e+00, align 8

define spir_func void @_Z5func3v() !dbg !27 {
  store double 1.000000e+00, double* @d, align 8, !dbg !28
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 6.0.0-1ubuntu2 (tags/RELEASE_600/final)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "warning2.cpp", directory: "/doesnt_matter")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!7 = distinct !DISubprogram(name: "func2", linkageName: "_Z5func2v", scope: !1, file: !1, line: 3, type: !8, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0)
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!11 = distinct !DISubprogram(name: "func1", linkageName: "_Z5func1v", scope: !1, file: !1, line: 2, type: !8, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0)
!12 = !DILocation(line: 10, column: 4, scope: !11)
!14 = distinct !DISubprogram(name: "kernel", scope: !1, file: !1, line: 1, type: !15, isLocal: false, isDefinition: true, scopeLine: 11, flags: DIFlagPrototyped, isOptimized: false, unit: !0)
!15 = !DISubroutineType(types: !16)
!16 = !{null}
!18 = !DILocation(line: 5, column: 3, scope: !14)
!19 = !DILocation(line: 6, column: 3, scope: !14)

!20 = !{i32 1}
!intel_types_that_use_aspects = !{!21}
!21 = !{!"MyStruct", i32 2}
!22 = !DILocalVariable(name: "tmp", scope: !7, file: !1, type: !23)
!23 = !DICompositeType(tag: DW_TAG_structure_type, name: "MyStruct", file: !1, line: 1, size: 4, elements: !25, identifier: "_ZTS6MyStruct")
!24 = !DILocation(line: 15, column: 3, scope: !7)
!25 = !{!26}
!26 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!27 = distinct !DISubprogram(name: "func3", linkageName: "_Z5func3v", scope: !1, file: !1, line: 3, type: !8, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0)
!28 = !DILocation(line: 20, column: 3, scope: !27)
