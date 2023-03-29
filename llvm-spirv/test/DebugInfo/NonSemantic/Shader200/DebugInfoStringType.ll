; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -spirv-text %t.bc -o %t.spt --spirv-debug-info-version=nonsemantic-shader-200
; RUN: FileCheck < %t.spt %s -check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -to-binary %t.spt -o %t.spv

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s -check-prefix=CHECK-LLVM

; CHECK-SPIRV: ExtInstImport [[#EISId:]] "NonSemantic.Shader.DebugInfo.200"
; CHECK-SPIRV: String [[#StrGreet:]] ".str.GREETING"
; CHECK-SPIRV: String [[#StrChar1:]] "CHARACTER_1"
; CHECK-SPIRV: String [[#StrChar2:]] "CHARACTER_2"
; CHECK-SPIRV: String [[#StrChar3:]] "CHARACTER_3"
; CHECK-SPIRV: TypeInt [[#TypeInt:]] 32 0
; CHECK-SPIRV: Constant [[#TypeInt]] [[#ConstZero:]] 0
; CHECK-SPIRV: Constant [[#TypeInt]] [[#Const80:]] 80
; CHECK-SPIRV: TypeVoid [[#TypeVoid:]]

; CHECK-SPIRV: [[#DINoneId:]] [[#EISId]] DebugInfoNone
; CHECK-SPIRV: [[#DataLocExpr:]] [[#EISId]] DebugExpression [[#]] [[#]] {{$}}
; CHECK-SPIRV: [[#LengthAddrExpr:]] [[#EISId]] DebugExpression [[#]] [[#]] {{$}}

; DebugTypeString NameId BaseTyId DataLocId SizeId LengthAddrId
; CHECK-SPIRV: [[#EISId]] DebugTypeString [[#StrGreet]] [[#DINoneId]] [[#DataLocExpr]] [[#ConstZero]] [[#LengthAddrExpr]]
; CHECK-SPIRV: [[#EISId]] DebugTypeString [[#StrChar1]] [[#DINoneId]] [[#DINoneId]] [[#Const80]] [[#DINoneId]]

; CHECK-SPIRV-COUNT-2: [[#LengthAddrVar:]] [[#EISId]] DebugLocalVariable
; CHECK-SPIRV-NEXT: [[#EISId]] DebugTypeString [[#StrChar2]] [[#DINoneId]] [[#DINoneId]] [[#ConstZero]] [[#LengthAddrVar]]
; CHECK-SPIRV-COUNT-3: [[#LengthAddrVar1:]] [[#EISId]] DebugLocalVariable
; CHECK-SPIRV-NEXT: [[#EISId]] DebugTypeString [[#StrChar3]] [[#DINoneId]] [[#DINoneId]] [[#ConstZero]] [[#LengthAddrVar1]]
; CHECK-SPIRV-COUNT-4: [[#LengthAddrVar2:]] [[#EISId]] DebugLocalVariable
; CHECK-SPIRV-NEXT: [[#EISId]] DebugTypeString [[#StrChar2]] [[#DINoneId]] [[#DINoneId]] [[#ConstZero]] [[#LengthAddrVar2]]

; CHECK-LLVM: !DICompileUnit(language: DW_LANG_Fortran95
; CHECK-LLVM-DAG: !DIStringType(name: "CHARACTER_1", size: 80)
; CHECK-LLVM-DAG: !DIStringType(name: ".str.GREETING", stringLengthExpression: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 8), stringLocationExpression: !DIExpression(DW_OP_push_object_address, DW_OP_deref))
; CHECK-LLVM-DAG: ![[#Scope:]] = distinct !DISubprogram(name: "print_greeting", linkageName: "print_greeting_"
; CHECK-LLVM-DAG: ![[#StrLenMD:]] = !DILocalVariable(name: "STRING1.len", scope: ![[#Scope]]
; CHECK-LLVM-DAG: !DIStringType(name: "CHARACTER_2", stringLength: ![[#StrLenMD]])
; CHECK-LLVM-DAG: ![[#StrLenMD1:]] = !DILocalVariable(name: "STRING2.len", scope: ![[#Scope]]
; CHECK-LLVM-DAG: !DIStringType(name: "CHARACTER_3", stringLength: ![[#StrLenMD1]])

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

%"QNCA_a0$i8*$rank0$" = type { ptr, i64, i64, i64, i64, i64 }

@strlit = internal unnamed_addr constant [5 x i8] c"HELLO"
@strlit.1 = internal unnamed_addr constant [3 x i8] c"TOM"
@"hello_world_$GREETING" = internal global %"QNCA_a0$i8*$rank0$" zeroinitializer, !dbg !2
@"hello_world_$NAME" = internal global [10 x i8] zeroinitializer, align 1, !dbg !10
@0 = internal unnamed_addr constant i32 65536, align 4
@1 = internal unnamed_addr constant i32 2, align 4
@strlit.2 = internal unnamed_addr constant [2 x i8] c", "

; Function Attrs: nounwind uwtable
define void @MAIN__() local_unnamed_addr #0 !dbg !4{
  %"hello_world_$GREETING_fetch.16" = load ptr, ptr @"hello_world_$GREETING", align 16, !dbg !20
  %fetch.15 = load i64, ptr getelementptr inbounds (%"QNCA_a0$i8*$rank0$", ptr @"hello_world_$GREETING", i64 0, i32 1), align 8, !dbg !20
  call void @llvm.dbg.value(metadata i64 %fetch.15, metadata !24, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.value(metadata i64 %fetch.15, metadata !31, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.value(metadata i64 10, metadata !28, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.value(metadata i64 10, metadata !32, metadata !DIExpression()), !dbg !21
  call void @llvm.dbg.declare(metadata ptr %"hello_world_$GREETING_fetch.16", metadata !26, metadata !DIExpression()), !dbg !36
  call void @llvm.dbg.declare(metadata ptr @"hello_world_$NAME", metadata !29, metadata !DIExpression()), !dbg !37
  ret void, !dbg !38
}

; Function Attrs: nofree nounwind uwtable
define void @print_greeting_(ptr noalias readonly %"print_greeting_$STRING1", ptr noalias readonly %"print_greeting_$STRING2", i64 %"STRING1.len$val", i64 %"STRING2.len$val") local_unnamed_addr #1 !dbg !22 {
alloca_1:
  call void @llvm.dbg.value(metadata i64 %"STRING1.len$val", metadata !24, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i64 %"STRING1.len$val", metadata !31, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i64 %"STRING2.len$val", metadata !28, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.value(metadata i64 %"STRING2.len$val", metadata !32, metadata !DIExpression()), !dbg !39
  call void @llvm.dbg.declare(metadata ptr %"print_greeting_$STRING1", metadata !26, metadata !DIExpression()), !dbg !40
  call void @llvm.dbg.declare(metadata ptr %"print_greeting_$STRING2", metadata !29, metadata !DIExpression()), !dbg !41
  ret void, !dbg !42
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { nounwind uwtable }
attributes #1 = { nofree nounwind uwtable}
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!18, !19}
!llvm.dbg.cu = !{!8}

!2 = !DIGlobalVariableExpression(var: !3, expr: !DIExpression())
!3 = distinct !DIGlobalVariable(name: "greeting", linkageName: "hello_world_$GREETING", scope: !4, file: !5, line: 3, type: !14, isLocal: true, isDefinition: true)
!4 = distinct !DISubprogram(name: "hello_world", linkageName: "MAIN__", scope: !5, file: !5, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !8, retainedNodes: !13)
!5 = !DIFile(filename: "hello.f90", directory: "/dev/null")
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !5, producer: "fortran", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !9, splitDebugInlining: false, nameTableKind: None)
!9 = !{!2, !10}
!10 = !DIGlobalVariableExpression(var: !11, expr: !DIExpression())
!11 = distinct !DIGlobalVariable(name: "name", linkageName: "hello_world_$NAME", scope: !4, file: !5, line: 2, type: !12, isLocal: true, isDefinition: true)
!12 = !DIStringType(name: "CHARACTER_1", size: 80)
!13 = !{}
!14 = !DIStringType(name: ".str.GREETING", stringLengthExpression: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 8), stringLocationExpression: !DIExpression(DW_OP_push_object_address, DW_OP_deref))
!18 = !{i32 2, !"Debug Info Version", i32 3}
!19 = !{i32 2, !"Dwarf Version", i32 4}
!20 = !DILocation(line: 6, column: 23, scope: !4)
!21 = !DILocation(line: 0, scope: !22, inlinedAt: !33)
!22 = distinct !DISubprogram(name: "print_greeting", linkageName: "print_greeting_", scope: !5, file: !5, line: 9, type: !6, scopeLine: 9, spFlags: DISPFlagDefinition, unit: !8, retainedNodes: !23)
!23 = !{!24, !26, !28, !29, !31, !32}
!24 = !DILocalVariable(name: "STRING1.len", scope: !22, type: !25, flags: DIFlagArtificial)
!25 = !DIBasicType(name: "INTEGER*8", size: 64, encoding: DW_ATE_signed)
!26 = !DILocalVariable(name: "string1", arg: 1, scope: !22, file: !5, line: 9, type: !27)
!27 = !DIStringType(name: "CHARACTER_2", stringLength: !24)
!28 = !DILocalVariable(name: "STRING2.len", scope: !22, type: !25, flags: DIFlagArtificial)
!29 = !DILocalVariable(name: "string2", arg: 2, scope: !22, file: !5, line: 9, type: !30)
!30 = !DIStringType(name: "CHARACTER_3", stringLength: !28)
!31 = !DILocalVariable(name: "_string1", arg: 3, scope: !22, type: !25, flags: DIFlagArtificial)
!32 = !DILocalVariable(name: "_string2", arg: 4, scope: !22, type: !25, flags: DIFlagArtificial)
!33 = distinct !DILocation(line: 0, scope: !34, inlinedAt: !35)
!34 = distinct !DISubprogram(name: "print_greeting_.t60p.t61p.t3v.t3v", linkageName: "print_greeting_.t60p.t61p.t3v.t3v", scope: !5, file: !5, type: !6, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !8, retainedNodes: !13, targetFuncName: "print_greeting_")
!35 = distinct !DILocation(line: 6, column: 8, scope: !4)
!36 = !DILocation(line: 9, column: 27, scope: !22, inlinedAt: !33)
!37 = !DILocation(line: 9, column: 36, scope: !22, inlinedAt: !33)
!38 = !DILocation(line: 7, column: 1, scope: !4)
!39 = !DILocation(line: 0, scope: !22)
!40 = !DILocation(line: 9, column: 27, scope: !22)
!41 = !DILocation(line: 9, column: 36, scope: !22)
!42 = !DILocation(line: 12, column: 1, scope: !22)
