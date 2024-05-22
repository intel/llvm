; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-allow-extra-diexpressions
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.rev.ll
; RUN: FileCheck %s --input-file %t.rev.ll

; RUN: llc -mtriple=%triple -dwarf-version=5 -filetype=obj -O0 < %t.rev.ll
; RUN: llc -mtriple=%triple -dwarf-version=4 -filetype=obj -O0 < %t.rev.ll

; RUN: llvm-spirv %t.bc -o %t.spv --spirv-debug-info-version=nonsemantic-shader-200
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.rev.ll
; RUN: FileCheck %s --input-file %t.rev.ll

; RUN: llc -mtriple=%triple -dwarf-version=5 -filetype=obj -O0 < %t.rev.ll
; RUN: llc -mtriple=%triple -dwarf-version=4 -filetype=obj -O0 < %t.rev.ll

; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-allow-extra-diexpressions --experimental-debuginfo-iterators=1
; RUN: llvm-spirv -r --experimental-debuginfo-iterators=1 %t.spv -o - | llvm-dis -o %t.rev.ll
; RUN: FileCheck %s --input-file %t.rev.ll

; RUN: llc -mtriple=%triple -dwarf-version=5 -filetype=obj -O0 < %t.rev.ll
; RUN: llc -mtriple=%triple -dwarf-version=4 -filetype=obj -O0 < %t.rev.ll

; RUN: llvm-spirv %t.bc -o %t.spv --spirv-debug-info-version=nonsemantic-shader-200 --experimental-debuginfo-iterators=1
; RUN: llvm-spirv -r --experimental-debuginfo-iterators=1 %t.spv -o - | llvm-dis -o %t.rev.ll
; RUN: FileCheck %s --input-file %t.rev.ll

; RUN: llc -mtriple=%triple -dwarf-version=5 -filetype=obj -O0 < %t.rev.ll
; RUN: llc -mtriple=%triple -dwarf-version=4 -filetype=obj -O0 < %t.rev.ll

; CHECK: DW_OP_constu, 42
; CHECK: DW_OP_plus_uconst, 42
; CHECK: DW_OP_plus
; CHECK: DW_OP_minus
; CHECK: DW_OP_mul
; CHECK: DW_OP_div
; CHECK: DW_OP_mod
; CHECK: DW_OP_or
; CHECK: DW_OP_and
; CHECK: DW_OP_xor
; CHECK: DW_OP_shl
; CHECK: DW_OP_shr
; CHECK: DW_OP_shra
; CHECK: DW_OP_deref
; CHECK: DW_OP_deref_size, 4
; CHECK: DW_OP_xderef
; CHECK: DW_OP_lit0
; CHECK: DW_OP_not
; CHECK: DW_OP_dup
; CHECK: DW_OP_regx, 1
; CHECK: DW_OP_bregx, 1, 4
; CHECK: DW_OP_push_object_address
; CHECK: DW_OP_swap
; CHECK: DW_OP_LLVM_convert, 8, DW_ATE_signed
; CHECK: DW_OP_stack_value

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: noinline nounwind uwtable
define dso_local signext i8 @foo(i8 signext %x) !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i8 %x, metadata !11, metadata !DIExpression()), !dbg !12
  call void @llvm.dbg.value(metadata i8 32, metadata !13, metadata !DIExpression(DW_OP_constu, 42, DW_OP_plus_uconst, 42, DW_OP_plus, DW_OP_minus, DW_OP_mul, DW_OP_div, DW_OP_mod, DW_OP_or, DW_OP_and, DW_OP_xor, DW_OP_shl, DW_OP_shr, DW_OP_shra, DW_OP_deref, DW_OP_deref_size, 4, DW_OP_xderef, DW_OP_lit0, DW_OP_not, DW_OP_dup, DW_OP_regx, 1, DW_OP_bregx, 1, 4, DW_OP_push_object_address, DW_OP_swap, DW_OP_LLVM_convert, 8, DW_ATE_signed, DW_OP_stack_value)), !dbg !15
  ret i8 %x, !dbg !16
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata)

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (trunk 353791) (llvm/trunk 353801)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "dbg.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "2a034da6937f5b9cf6dd2d89127f57fd")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 9.0.0 (trunk 353791) (llvm/trunk 353801)"}
!7 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !8, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "signed char", size: 8, encoding: DW_ATE_signed_char)
!11 = !DILocalVariable(name: "x", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!12 = !DILocation(line: 1, column: 29, scope: !7)
!13 = !DILocalVariable(name: "y", scope: !7, file: !1, line: 3, type: !14)
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DILocation(line: 3, column: 14, scope: !7)
!16 = !DILocation(line: 4, column: 3, scope: !7)
