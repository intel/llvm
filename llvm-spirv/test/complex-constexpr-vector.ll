; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

define linkonce_odr hidden spir_func void @foo() {
entry:

; CHECK-SPIRV: Constant [[#]] [[#CONSTANT1:]] 65793

; 81 is OpCompositeExtract opcode
; CHECK-SPIRV: SpecConstantOp [[#]] [[#MEMBER1_EXTRACT1:]] 81 [[#MEMBER1_EXTRACT1_BITCAST:]] 0
; CHECK-SPIRV: Constant [[#]] [[#CONSTANT2:]] 131586
; CHECK-SPIRV: SpecConstantOp [[#]] [[#MEMBER1_EXTRACT2:]] 81 [[#MEMBER1_EXTRACT2_BITCAST:]] 0
; 128 is OpIAdd opcode
; CHECK-SPIRV: SpecConstantOp [[#]] [[#MEMBER_1:]] 128 [[#MEMBER1_EXTRACT1]] [[#MEMBER1_EXTRACT2]]

; CHECK-SPIRV: SpecConstantOp [[#]] [[#MEMBER2_EXTRACT1:]] 81 [[#MEMBER1_EXTRACT1_BITCAST]] 1
; CHECK-SPIRV: SpecConstantOp [[#]] [[#MEMBER2_EXTRACT2:]] 81 [[#MEMBER1_EXTRACT2_BITCAST]] 1
; CHECK-SPIRV: SpecConstantOp [[#]] [[#MEMBER_2:]] 128 [[#MEMBER2_EXTRACT1]] [[#MEMBER2_EXTRACT2]]

; CHECK-SPIRV: SpecConstantOp [[#]] [[#MEMBER3_EXTRACT1:]] 81 [[#MEMBER1_EXTRACT1_BITCAST]] 2
; CHECK-SPIRV: SpecConstantOp [[#]] [[#MEMBER3_EXTRACT2:]] 81 [[#MEMBER1_EXTRACT2_BITCAST]] 2
; CHECK-SPIRV: SpecConstantOp [[#]] [[#MEMBER_3:]] 128 [[#MEMBER3_EXTRACT1]] [[#MEMBER3_EXTRACT2]]
; CHECK-SPIRV: Undef [[#]] [[#MEMBER_4:]]

; CHECK-SPIRV: ConstantComposite [[#]] [[#COMPOS1:]] [[#CONSTANT1]] [[#CONSTANT1]]
; 124 is OpBitcast opcode
; CHECK-SPIRV: SpecConstantOp [[#]] [[#MEMBER1_EXTRACT1_BITCAST]] 124 [[#COMPOS1]]

; CHECK-SPIRV: ConstantComposite [[#]] [[#COMPOS2:]] [[#CONSTANT2]] [[#CONSTANT2]]
; CHECK-SPIRV: SpecConstantOp [[#]] [[#MEMBER1_EXTRACT2_BITCAST]] 124 [[#COMPOS2]]

; CHECK-SPIRV: ConstantComposite [[#]] [[#FINAL_COMPOS:]] [[#MEMBER_1]] [[#MEMBER_2]] [[#MEMBER_3]] [[#MEMBER_4]]
; CHECK-SPIRV: DebugValue [[#]] [[#FINAL_COMPOS]]


; CHECK-LLVM: #dbg_value(
; CHECK-LLVM-SAME:   <4 x i8> <
; CHECK-LLVM-SAME:   i8 add (
; CHECK-LLVM-SAME:     i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 65793, i32 65793> to <8 x i8>), i32 0),
; CHECK-LLVM-SAME:     i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 131586, i32 131586> to <8 x i8>), i32 0)),
; CHECK-LLVM-SAME:   i8 add (
; CHECK-LLVM-SAME:     i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 65793, i32 65793> to <8 x i8>), i32 1),
; CHECK-LLVM-SAME:     i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 131586, i32 131586> to <8 x i8>), i32 1)),
; CHECK-LLVM-SAME:   i8 add (
; CHECK-LLVM-SAME:     i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 65793, i32 65793> to <8 x i8>), i32 2),
; CHECK-LLVM-SAME:     i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 131586, i32 131586> to <8 x i8>), i32 2)),
; CHECK-LLVM-SAME:   i8 undef>,
; CHECK-LLVM-SAME:   ![[#]], !DIExpression(), ![[#]])
  call void @llvm.dbg.value(
    metadata <4 x i8> <
    i8 add (
      i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 65793, i32 65793> to <8 x i8>), i32 0),
      i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 131586, i32 131586> to <8 x i8>), i32 0)),
    i8 add (
      i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 65793, i32 65793> to <8 x i8>), i32 1),
      i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 131586, i32 131586> to <8 x i8>), i32 1)),
    i8 add (
      i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 65793, i32 65793> to <8 x i8>), i32 2),
      i8 extractelement (<8 x i8> bitcast (<2 x i32> <i32 131586, i32 131586> to <8 x i8>), i32 2)),
    i8 undef>,
      metadata !12, metadata !DIExpression()), !dbg !7
  ret void
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 13.0.0 (https://github.com/intel/llvm.git)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "main.cpp", directory: "/export/users")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{!"clang version 13.0.0"}
!6 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 1, type: !8, scopeLine: 4, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DILocation(line: 1, scope: !6, inlinedAt: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{!10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 1, column: 0, scope: !6)
!12 = !DILocalVariable(name: "resVec", scope: !6, file: !1, line: 1, type: !13)
!13 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "vec<cl::sycl::detail::half_impl::half, 3>", scope: !6, file: !1, line: 1, size: 64, flags: DIFlagTypePassByValue, elements: !2)
