; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -s %t.bc -o - | llvm-dis -o - | FileCheck %s
; RUN: llvm-spirv %t.bc

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind ssp uwtable
define void @foo() #0 !dbg !4 {
entry:
  call void @llvm.dbg.value(
    metadata <3 x i8> <
      i8 add (
        i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 65793, i32 65793> to <8 x i8>), i32 0),
        i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 131586, i32 131586> to <8 x i8>), i32 0)),
      i8 add (
        i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 65793, i32 65793> to <8 x i8>), i32 1),
        i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 131586, i32 131586> to <8 x i8>), i32 1)),
      i8 add (
        i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 65793, i32 65793> to <8 x i8>), i32 2),
        i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 131586, i32 131586> to <8 x i8>), i32 2))>,
      metadata !12, metadata !DIExpression()), !dbg !18
; CHECK: call void @llvm.dbg.value(
; CHECK-SAME:   metadata <3 x i8> <
; CHECK-SAME:     i8 add (
; CHECK-SAME:       i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 65793, i32 65793> to <8 x i8>), i32 0),
; CHECK-SAME:       i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 131586, i32 131586> to <8 x i8>), i32 0)),
; CHECK-SAME:     i8 add (
; CHECK-SAME:       i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 65793, i32 65793> to <8 x i8>), i32 1),
; CHECK-SAME:       i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 131586, i32 131586> to <8 x i8>), i32 1)),
; CHECK-SAME:     i8 add (
; CHECK-SAME:       i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 65793, i32 65793> to <8 x i8>), i32 2),
; CHECK-SAME:       i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 131586, i32 131586> to <8 x i8>), i32 2))>,
; CHECK-SAME:     metadata ![[#]], metadata !DIExpression()), !dbg ![[#]]
  call void @llvm.dbg.value(
    metadata <4 x half> <
    half fadd (
      half extractelement (<4 x half> bitcast (<2 x i32> <i32 65793, i32 65793> to <4 x half>), i32 0),
      half extractelement (<4 x half> bitcast (<2 x i32> <i32 131586, i32 131586> to <4 x half>), i32 0)),
    half fadd (
      half extractelement (<4 x half> bitcast (<2 x i32> <i32 65793, i32 65793> to <4 x half>), i32 1),
      half extractelement (<4 x half> bitcast (<2 x i32> <i32 131586, i32 131586> to <4 x half>), i32 1)),
    half fadd (
      half extractelement (<4 x half> bitcast (<2 x i32> <i32 65793, i32 65793> to <4 x half>), i32 2),
      half extractelement (<4 x half> bitcast (<2 x i32> <i32 131586, i32 131586> to <4 x half>), i32 2)),
    half undef>,
    metadata !23, metadata !DIExpression()), !dbg !20
; CHECK: call void @llvm.dbg.value(
; CHECK-SAME:   metadata <4 x half> <
; CHECK-SAME:   half fadd (
; CHECK-SAME:     half extractelement (<4 x half> bitcast (<2 x i32> <i32 65793, i32 65793> to <4 x half>), i32 0),
; CHECK-SAME:     half extractelement (<4 x half> bitcast (<2 x i32> <i32 131586, i32 131586> to <4 x half>), i32 0)),
; CHECK-SAME:   half fadd (
; CHECK-SAME:     half extractelement (<4 x half> bitcast (<2 x i32> <i32 65793, i32 65793> to <4 x half>), i32 1),
; CHECK-SAME:     half extractelement (<4 x half> bitcast (<2 x i32> <i32 131586, i32 131586> to <4 x half>), i32 1)),
; CHECK-SAME:   half fadd (
; CHECK-SAME:     half extractelement (<4 x half> bitcast (<2 x i32> <i32 65793, i32 65793> to <4 x half>), i32 2),
; CHECK-SAME:     half extractelement (<4 x half> bitcast (<2 x i32> <i32 131586, i32 131586> to <4 x half>), i32 2)),
; CHECK-SAME:   half undef>,
; CHECK-SAME:   metadata ![[#]], metadata !DIExpression()), !dbg ![[#]]
  ret void, !dbg !22
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind ssp uwtable  }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0, !26}
!llvm.module.flags = !{!13, !14, !15}
!llvm.ident = !{!16}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.7.0 (trunk 235110) (llvm/trunk 235108)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "t.c", directory: "/path/to/dir")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 2, type: !8, isLocal: true, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "a", arg: 1, scope: !7, file: !1, line: 2, type: !10)
!13 = !{i32 2, !"Dwarf Version", i32 2}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"PIC Level", i32 2}
!16 = !{!"clang version 3.7.0 (trunk 235110) (llvm/trunk 235108)"}
!17 = !DIExpression()
!18 = !DILocation(line: 2, column: 52, scope: !7, inlinedAt: !19)
!19 = distinct !DILocation(line: 4, column: 3, scope: !4)
!20 = !DILocation(line: 5, scope: !4, inlinedAt: !21)
!21 = !DILocation(line: 1, column: 0, scope: !4)
!22 = !DILocation(line: 6, column: 1, scope: !4)
!23 = !DILocalVariable(name: "resVec", scope: !4, file: !24, line: 1, type: !25)
!24 = !DIFile(filename: "main.cpp", directory: "/export/users")
!25 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "vec<cl::sycl::detail::half_impl::half, 3>", scope: !4, file: !24, line: 1, size: 64, flags: DIFlagTypePassByValue, elements: !2)
!26 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !24, producer: "clang version 13.0.0 (https://github.com/intel/llvm.git)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
