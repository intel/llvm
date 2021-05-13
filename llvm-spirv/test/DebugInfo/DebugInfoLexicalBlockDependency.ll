; This test checks the translation of debug info is correct when:
;   - Subprogram contains a lexical block LB and a local variable VAL
;   - The parent scope of the local variable VAL is the lexical block LB
;   - The parent scope of the lexical block LB is the subprogram.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck %s --input-file %t.rev.ll

target triple = "spir64-unknown-unknown"

; Function Attrs: noinline nounwind
define spir_kernel void @test(i64 %value) #0 !dbg !9 {
entry:
  %value.addr = alloca i32, align 1, !dbg !17
  call void @llvm.dbg.declare(metadata i32* %value.addr, metadata !13, metadata !DIExpression()), !dbg !18
  %value.trunc = trunc i64 %value to i32, !dbg !17
  store i32 %value.trunc, i32* %value.addr, align 4, !dbg !17
  ret void
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}
!spirv.MemoryModel = !{!5}
!spirv.Source = !{!6}
!opencl.spir.version = !{!7}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!4}
!opencl.used.optional.core.features = !{!4}
!spirv.Generator = !{!8}

!0 = !{i32 7, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "spirv", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, imports: !4)
!3 = !DIFile(filename: "1", directory: "/")
!4 = !{}
!5 = !{i32 2, i32 2}
!6 = !{i32 4, i32 200000}
!7 = !{i32 2, i32 0}
!8 = !{i16 6, i16 14}
!9 = distinct !DISubprogram(name: "test", scope: null, file: !3, line: 13, type: !10, scopeLine: 13, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2, templateParams: !4, retainedNodes: !12)
!10 = !DISubroutineType(types: !11)
!11 = !{null}
!12 = !{!13}
!13 = !DILocalVariable(name: "value", scope: !14, file: !3, line: 12, type: !15)
!14 = distinct !DILexicalBlock(scope: !9, file: !3, line: 13, column: 1)
!15 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !16)
!16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!17 = !DILocation(line: 13, column: 1, scope: !14)
!18 = !DILocation(line: 12, column: 23, scope: !14)

; Debug location metadata id of original instruction.
; CHECK: %value.addr = alloca i32, align 1, !dbg ![[#DL:]]

; The function contains the local variable VAL in retainedNodes.
; CHECK: ![[#SP:]] = distinct !DISubprogram(name: "test"
; CHECK-SAME: retainedNodes: ![[#NODES:]]
; CHECK: ![[#NODES]] = !{![[#VAL:]]}

; The local variable VAL lies in block LB, whose parent scope is the subprogram.
; CHECK: ![[#VAL]] = !DILocalVariable(name: "value", scope: ![[#LB:]]
; CHECK: ![[#LB]] = distinct !DILexicalBlock(scope: ![[#SP]]

; The debug location should be attached to block LB.
; CHECK: ![[#DL]] = !DILocation(
; CHECK-SAME: scope: ![[#LB]]

; Other lexical blocks are unexpected.
; CHECK-NOT: distinct !DILexicalBlock
