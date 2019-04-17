; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv -spirv-mem2reg=false
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o %t.ll

; RUN: llc -mtriple=x86_64-unknown-unknown -o - %t.ll | FileCheck %s
; RUN: llc -mtriple=x86_64-unknown-unknown -filetype=obj < %t.ll \
; RUN:   | llvm-dwarfdump -v - | FileCheck %s --check-prefix=DWARF

define i1 @test() !dbg !4 {
entry:
  %end = alloca i64, align 8
  br label %while.cond

while.cond:
  call void @llvm.dbg.value(metadata i64* %end, metadata !5, metadata !6), !dbg !7
  %call = call i1 @fn(i64* %end, i64* %end, i64* null, i8* null, i64 0, i64* null, i32* null, i8* null), !dbg !7
  br label %while.body

while.body:
  br i1 0, label %while.end, label %while.cond

while.end:
  ret i1 true
}

; CHECK-LABEL: test
; CHECK:       #DEBUG_VALUE: test:w <- [DW_OP_plus_uconst 8, DW_OP_deref] $rsp
; DWARF:  DW_AT_location [DW_FORM_sec_offset] (
; DWARF-NEXT:   {{.*}}, {{.*}}: DW_OP_breg7 RSP+8)

declare i1 @fn(i64*, i64*, i64*, i8*, i64, i64*, i32*, i8*)
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2,!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 4.0.0", emissionKind: FullDebug)
!1 = !DIFile(filename: "test.c", directory: "/")
!2 = !{i32 2, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "test", type: !10, unit: !0)
!5 = !DILocalVariable(name: "w", scope: !4, type: !9)
!6 = !DIExpression(DW_OP_deref)
!7 = !DILocation(line: 210, column: 12, scope: !4)
!8 = !{!9}
!9 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!10 = !DISubroutineType(types: !8)
target triple = "spir64-unknown-unknown"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
