; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -spirv-text %t.bc -o %t.spt --spirv-debug-info-version=nonsemantic-shader-200
; RUN: FileCheck < %t.spt %s -check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -to-binary %t.spt -o %t.spv

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s -check-prefix=CHECK-LLVM

; CHECK-SPIRV: ExtInstImport [[#EISId:]] "NonSemantic.Shader.DebugInfo.200"

; CHECK-SPIRV: String [[#LocalVarNameId:]] "A$1$upperbound"
; CHECK-SPIRV-DAG: TypeInt [[#TyInt32Id:]] 32 0
; CHECK-SPIRV-DAG: TypeInt [[#TyInt64Id:]] 64 0
; CHECK-NOT: THIS LINE IS USED TO SEPARATE DAGs
; CHECK-SPIRV-DAG: Constant [[#TyInt32Id]] [[#Constant15Id:]] 15{{[[:space:]]}}
; CHECK-SPIRV-DAG: Constant [[#TyInt32Id]] [[#Constant24Id:]] 24{{[[:space:]]}}
; CHECK-SPIRV-DAG: Constant [[#TyInt32Id]] [[#Constant25Id:]] 25{{[[:space:]]}}
; CHECK-SPIRV-DAG: Constant [[#TyInt32Id]] [[#Constant27Id:]] 27{{[[:space:]]}}
; CHECK-SPIRV-DAG: Constant [[#TyInt32Id]] [[#Constant33Id:]] 33{{[[:space:]]}}
; CHECK-SPIRV-DAG: Constant [[#TyInt32Id]] [[#Constant34Id:]] 34{{[[:space:]]}}
; CHECK-SPIRV-DAG: Constant [[#TyInt32Id]] [[#Constant67Id:]] 67{{[[:space:]]}}
; CHECK-SPIRV-DAG: Constant [[#TyInt32Id]] [[#Constant68Id:]] 68{{[[:space:]]}}
; CHECK-SPIRV-DAG: Constant [[#TyInt64Id]] [[#Constant1Id:]] 1 0
; CHECK-SPIRV-DAG: Constant [[#TyInt64Id]] [[#Constant1000Id:]] 1000 0
; CHECK-SPIRV: [[#DINoneId:]] [[#EISId]] DebugInfoNone

; CHECK-SPIRV: [[#DebugFuncId:]] [[#EISId]] DebugFunction
; CHECK-SPIRV: [[#LocalVarId:]] [[#EISId]] DebugLocalVariable [[#LocalVarNameId]] [[#]] [[#]] [[#]] [[#]] [[#DebugFuncId]]
; CHECK-SPIRV: [[#DebugTypeTemplate:]] [[#EISId]] DebugTypeTemplate [[#DebugFuncId]]
; CHECK-SPIRV: [[#EISId]] DebugTypeSubrange [[#Constant1Id]] [[#LocalVarId]]  [[#DINoneId]] {{$}}

; CHECK-SPIRV: [[#DIExprId:]] [[#EISId]] DebugExpression
; CHECK-SPIRV: [[#EISId]] DebugTypeSubrange [[#DIExprId]] [[#DIExprId]] [[#DINoneId]] {{$}}

; CHECK-SPIRV: [[#EISId]] DebugTypeSubrange [[#Constant1Id]] [[#DINoneId]] [[#Constant1000Id]] {{$}}

; CHECK-SPIRV: [[#EISId]] DebugLine [[#]] [[#Constant15Id]] [[#Constant15Id]] [[#Constant67Id]] [[#Constant68Id]]
; CHECK-SPIRV: [[#EISId]] DebugLine [[#]] [[#Constant27Id]] [[#Constant27Id]] [[#Constant24Id]] [[#Constant25Id]]
; CHECK-SPIRV: [[#EISId]] DebugLine [[#]] [[#Constant34Id]] [[#Constant34Id]] [[#Constant33Id]] [[#Constant34Id]]

; CHECK-LLVM: ![[#Scope_A:]] = distinct !DISubprogram(name: "random_fill_sp.DIR.OMP.TARGET.8.split.split.split.split"
; CHECK-LLVM: [[#Subrange1:]] = !DISubrange(lowerBound: 1, upperBound: ![[#UpperBound:]])
; CHECK-LLVM: [[#UpperBound]] = !DILocalVariable(name: "A$1$upperbound"
; CHECK-LLVM: !DILocation(line: 15, column: 67, scope: ![[#Scope_A]]
; CHECK-LLVM: ![[#Scope_B:]] = distinct !DISubprogram(name: "random_fill_sp.DIR.OMP.TARGET.8.split.split.split.split"
; CHECK-LLVM: !DISubrange(lowerBound: !DIExpression(), upperBound: !DIExpression())
; CHECK-LLVM: !DILocation(line: 15, column: 67, scope: ![[#Scope_B]]
; CHECK-LLVM: ![[#Scope_C:]] = distinct !DISubprogram(name: "test_target_map_array_default_IP_test_array_map_no_map_type_.DIR.OMP.TARGET.340.split"
; CHECK-LLVM: !DISubrange(count: 1000, lowerBound: 1)
; CHECK-LLVM: !DILocation(line: 27, column: 24, scope: ![[#Scope_C]]
; CHECK-LLVM: ![[#Scope_D:]] = distinct !DISubprogram(name: "test"
; CHECK-LLVM: !DILocation(line: 34, column: 33, scope: ![[#Scope_D]]

; ModuleID = 'DebugInfoSubrangeUpperBound.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

%structtype = type { [72 x i1] }
%"QNCA_a0$float" = type { ptr addrspace(4), i64, i64, i64, i64, i64, [1 x %structtype2] }
%structtype2 = type { i64, i64, i64 }

; Function Attrs: noinline nounwind
define spir_kernel void @__omp_offloading_811_198142f_random_fill_sp_l25(ptr addrspace(1) noalias %0, ptr byval(%structtype) %"ascast$val", ptr addrspace(1) noalias %"ascastB$val") #0 !kernel_arg_addr_space !9 !kernel_arg_access_qual !10 !kernel_arg_type !11 !kernel_arg_type_qual !12 !kernel_arg_base_type !11 {
newFuncRoot:
  call void @llvm.dbg.value(metadata ptr %"ascast$val", metadata !13, metadata !DIExpression(DW_OP_deref)), !dbg !27
  call void @llvm.dbg.value(metadata ptr %"ascast$val", metadata !28, metadata !DIExpression(DW_OP_deref)), !dbg !42
  call void @llvm.dbg.value(metadata ptr addrspace(1) %"ascastB$val", metadata !47, metadata !DIExpression(DW_OP_deref)), !dbg !51
  call void @llvm.dbg.value(metadata ptr addrspace(1) %0, metadata !54, metadata !DIExpression(DW_OP_deref)), !dbg !59
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
!9 = !{i32 0, i32 0, i32 0}
!10 = !{!"none", !"none", !"none"}
!11 = !{!"structtype", !"structtype", !"structtype"}
!12 = !{!"", !"", !""}
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
!51 = !DILocation(line: 27, column: 24, scope: !46)
!52 = distinct !DISubprogram(name: "test", scope: !3, file: !3, line: 51, type: !53, scopeLine: 51, flags: DIFlagArtificial, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2)
!53 = !DISubroutineType(types: !7)
!54 = !DILocalVariable(name: "isHost", scope: !52, file: !3, line: 34, type: !55)
!55 = !DICompositeType(tag: DW_TAG_array_type, baseType: !56, elements: !57)
!56 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!57 = !{!58}
!58 = !DISubrange(count: -1)
!59 = !DILocation(line: 34, column: 33, scope: !52)
