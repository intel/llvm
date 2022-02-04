; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -spirv-text %t.bc -o %t.spt
; RUN: FileCheck < %t.spt %s -check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -to-binary %t.spt -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s -check-prefix=CHECK-LLVM

; CHECK-SPIRV: String [[#VarNameId:]] "A$1$upperbound"
; CHECK-SPIRV: [[#FuncNameId:]] "random_fill_sp"
; CHECK-SPIRV: TypeInt [[#TypeIntId:]] 64 0
; CHECK-SPIRV: Constant [[#TypeIntId]] [[#LowerBoundId:]] 1 0
; CHECK-SPIRV: [[#DbgFuncId:]] [[#]] DebugFunction [[#FuncNameId]]
; CHECK-SPIRV: [[#DbgTemplateId:]] [[#]] DebugTemplate [[#DbgFuncId]]
; CHECK-SPIRV: [[#]] [[#DbgLocVarId:]] [[#]] DebugLocalVariable [[#VarNameId]] [[#]] [[#]] [[#]] [[#]] [[#DbgTemplateId]]
; CHECK-SPIRV: DebugTypeArray [[#]] [[#DbgLocVarId]] [[#LowerBoundId]]

; CHECK-SPIRV: [[#DbgExprId:]] [[#]] DebugExpression
; CHECK-SPIRV: DebugTypeArray [[#]] [[#DbgExprId]] [[#DbgExprId]]

; CHECK-LLVM: !DICompositeType(tag: DW_TAG_array_type, baseType: ![[#BaseType:]], size: 32, elements: ![[#Subrange1:]])
; CHECK-LLVM: [[#BaseType]] = !DIBasicType(name: "REAL*4", size: 32, encoding: DW_ATE_float)
; CHECK-LLVM: [[#Subrange1]] = !{![[#Subrange2:]]}
; CHECK-LLVM: [[#Subrange2:]] = !DISubrange(lowerBound: 1, upperBound: ![[#UpperBound:]])
; CHECK-LLVM: [[#UpperBound]] = !DILocalVariable(name: "A$1$upperbound"

; CHECK-LLVM: !DICompositeType(tag: DW_TAG_array_type, baseType: ![[#]], size: 32, elements: ![[#SubrangeExpr1:]])
; CHECK-LLVM: [[#SubrangeExpr1]] = !{![[#SubrangeExpr2:]]}
; CHECK-LLVM: ![[#SubrangeExpr2]] = !DISubrange(lowerBound: !DIExpression(), upperBound: !DIExpression())

; CHECK-LLVM: !DISubrange(count: 1000, lowerBound: 1)

; ModuleID = 'DebugInfoSubrangeUpperBound.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

%structtype = type { [72 x i1] }
%"QNCA_a0$float" = type { float addrspace(4)*, i64, i64, i64, i64, i64, [1 x %structtype2] }
%structtype2 = type { i64, i64, i64 }

; Function Attrs: noinline nounwind
define spir_kernel void @__omp_offloading_811_198142f_random_fill_sp_l25(%structtype* byval(%structtype) %"ascast$val", [1000 x i32] addrspace(1)* noalias %"ascastB$val") #0 !kernel_arg_addr_space !9 !kernel_arg_access_qual !10 !kernel_arg_type !11 !kernel_arg_type_qual !12 !kernel_arg_base_type !11 {
newFuncRoot:
  %.ascast = bitcast %structtype* %"ascast$val" to %"QNCA_a0$float"*
  call void @llvm.dbg.value(metadata %"QNCA_a0$float"* %.ascast, metadata !13, metadata !DIExpression(DW_OP_deref)), !dbg !27
  call void @llvm.dbg.value(metadata %"QNCA_a0$float"* %.ascast, metadata !28, metadata !DIExpression(DW_OP_deref)), !dbg !42
  call void @llvm.dbg.value(metadata [1000 x i32] addrspace(1)* %"ascastB$val", metadata !47, metadata !DIExpression(DW_OP_deref)), !dbg !56
  ret void
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}
!spirv.MemoryModel = !{!4}
!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!5}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!6}
!opencl.used.extensions = !{!7}
!opencl.used.optional.core.features = !{!7}
!spirv.Generator = !{!8}

!0 = !{i32 7, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_OpenCL, file: !3, producer: "Fortran", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!3 = !DIFile(filename: "f.f90", directory: "Fortran")
!4 = !{i32 2, i32 2}
!5 = !{i32 4, i32 200000}
!6 = !{i32 2, i32 0}
!7 = !{}
!8 = !{i16 6, i16 14}
!9 = !{i32 0}
!10 = !{!"none"}
!11 = !{!"structtype"}
!12 = !{!""}
!13 = !DILocalVariable(name: "a", scope: !14, file: !3, line: 15, type: !18)
!14 = distinct !DISubprogram(name: "random_fill_sp.DIR.OMP.TARGET.8.split.split.split.split", scope: null, file: !3, line: 25, type: !15, scopeLine: 25, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, templateParams: !7, retainedNodes: !17)
!15 = !DISubroutineType(types: !16)
!16 = !{null}
!17 = !{!13}
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !19, size: 64)
!19 = !DICompositeType(tag: DW_TAG_array_type, baseType: !20, size: 32, elements: !21)
!20 = !DIBasicType(name: "REAL*4", size: 32, encoding: DW_ATE_float)
!21 = !{!22}
!22 = !DISubrange(lowerBound: 1, upperBound: !23)
!23 = !DILocalVariable(name: "A$1$upperbound", scope: !24, type: !26, flags: DIFlagArtificial)
!24 = distinct !DISubprogram(name: "random_fill_sp", linkageName: "random_fill_sp", scope: null, file: !3, line: 15, type: !15, scopeLine: 15, spFlags: DISPFlagDefinition, unit: !2, templateParams: !7, retainedNodes: !25)
!25 = !{!23}
!26 = !DIBasicType(name: "INTEGER*8", size: 64, encoding: DW_ATE_signed)
!27 = !DILocation(line: 15, column: 67, scope: !14)
!28 = !DILocalVariable(name: "a", scope: !29, file: !3, line: 15, type: !33)
!29 = distinct !DISubprogram(name: "random_fill_sp.DIR.OMP.TARGET.8.split.split.split.split", scope: null, file: !3, line: 25, type: !30, scopeLine: 25, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, templateParams: !7, retainedNodes: !32)
!30 = !DISubroutineType(types: !31)
!31 = !{null}
!32 = !{!28}
!33 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !34, size: 64)
!34 = !DICompositeType(tag: DW_TAG_array_type, baseType: !35, size: 32, elements: !36)
!35 = !DIBasicType(name: "REAL*4", size: 32, encoding: DW_ATE_float)
!36 = !{!37}
!37 = !DISubrange(lowerBound: !DIExpression(), upperBound: !DIExpression())
!38 = !DILocalVariable(name: "A$1$upperbound", scope: !39, type: !41, flags: DIFlagArtificial)
!39 = distinct !DISubprogram(name: "random_fill_sp", linkageName: "random_fill_sp", scope: null, file: !3, line: 15, type: !30, scopeLine: 15, spFlags: DISPFlagDefinition, unit: !2, templateParams: !7, retainedNodes: !40)
!40 = !{!38}
!41 = !DIBasicType(name: "INTEGER*8", size: 64, encoding: DW_ATE_signed)
!42 = !DILocation(line: 15, column: 67, scope: !29)
!43 = !DIBasicType(name: "INTEGER*4", size: 32, encoding: DW_ATE_signed)
!44 = !{}
!45 = !DISubroutineType(types: !44)
!46 = distinct !DISubprogram(name: "test_target_map_array_default_IP_test_array_map_no_map_type_.DIR.OMP.TARGET.340.split", scope: !3, file: !3, line: 32, type: !45, scopeLine: 32, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!47 = !DILocalVariable(name: "compute_array", scope: !46, file: !3, line: 27, type: !48)
!48 = !DICompositeType(tag: DW_TAG_array_type, baseType: !43, elements: !49)
!49 = !{!50}
!50 = !DISubrange(count: 1000, lowerBound: 1)
!56 = !DILocation(line: 27, column: 24, scope: !46)
