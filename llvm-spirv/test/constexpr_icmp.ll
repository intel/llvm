; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: llvm-spirv -r -emit-opaque-pointers %t.spv -o %t.bc
; RUN: llvm-dis %t.bc -o %t.ll
; RUN: FileCheck %s --input-file %t.spt -check-prefix=CHECK-SPIRV
; RUN: FileCheck %s --input-file %t.ll  -check-prefix=CHECK-LLVM

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

define linkonce_odr hidden spir_func void @foo() {
entry:
; CHECK-SPIRV: [[#]] ExtInst [[#]] [[#]] [[#]] DebugValue [[#]] [[#]] [[#]]
; CHECK-LLVM: call void @llvm.dbg.value(metadata <2 x i8> <i8 zext (i1 icmp ne (i8 extractelement (<16 x i8> bitcast (<2 x i64> <i64 72340172838076673, i64 72340172838076673> to <16 x i8>), i32 0), i8 0) to i8), i8 zext (i1 icmp ne (i8 extractelement (<16 x i8> bitcast (<2 x i64> <i64 72340172838076673, i64 72340172838076673> to <16 x i8>), i32 0), i8 0) to i8)>, metadata ![[#]], metadata !DIExpression()), !dbg ![[#]]

  call void @llvm.dbg.value(
    metadata <2 x i8> <i8 zext (i1 icmp ne (i8 extractelement (<16 x i8> bitcast (<2 x i64> <i64 72340172838076673, i64 72340172838076673> to <16 x i8>), i64 0), i8 0) to i8),
                       i8 zext (i1 icmp ne (i8 extractelement (<16 x i8> bitcast (<2 x i64> <i64 72340172838076673, i64 72340172838076673> to <16 x i8>), i64 0), i8 0) to i8)>,
    metadata !12,
    metadata !DIExpression()), !dbg !7
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
