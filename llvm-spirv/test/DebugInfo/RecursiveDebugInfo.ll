; Test checks that the translator converts debug info correctly and
; doesn't crash if the module has recursive template parameters definition.

; This LLVM IR was generated using Intel SYCL Clang compiler (https://github.com/intel/llvm)

; recursive_debug_info.cpp:
;
;  namespace s = cl::sycl;
;
;  template<typename Container>
;  struct iterator {
;    int *Itr;
;    iterator() : Itr(nullptr) {}
;    iterator(int* itr) : Itr(itr) {}
;  };
;
;  struct vector {
;    int *Start;
;    typedef iterator<vector> vec_it;
;    vec_it begin() {
;      return vec_it(Start);
;    }
;  };
;
;  class foo;
;
;  int main() {
;    s::queue q;
;    auto e = q.submit([=](s::handler &cgh) {
;      cgh.single_task<class foo>([=]() {
;        iterator<vector> IV;
;        vector V;
;      });
;    });
;    e.wait();
;    return 0;
;  }

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s

; CHECK: [[IT_VEC:![0-9]+]] = !DICompositeType(tag: DW_TAG_structure_type, name: "iterator<vector>", {{.+}}, templateParams: [[TMPL_P:![0-9]+]]
; CHECK: [[TMPL_P]] = !{[[TMPL_P1:![0-9]+]]}
; CHECK: [[TMPL_P1]] = !DITemplateTypeParameter(name: "Container", type: [[CTNR_TY:![0-9]+]])
; CHECK: [[CTNR_TY]] = !DICompositeType(tag: DW_TAG_structure_type, name: "vector", {{.+}}, elements: [[ELMS:![0-9]+]]
; CHECK: [[ELMS]] = !{!{{[0-9]+}}, [[EL2:![0-9]+]]}
; CHECK: [[EL2]] = !DISubprogram(name: "begin", {{.+}}, type: [[SPRG_TY:![0-9]+]]
; CHECK: [[SPRG_TY]] = !DISubroutineType(types: [[FNC_TYS:![0-9]+]])
; CHECK: [[FNC_TYS]] = !{[[FNC_TY1:![0-9]+]], !{{[0-9]+}}}
; CHECK: [[FNC_TY1]] = !DIDerivedType(tag: DW_TAG_typedef, name: "vec_it", {{.+}}, baseType: [[IT_VEC]])

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

%"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" = type { i8 }
%struct._ZTS8iteratorI6vectorE.iterator = type { i32 addrspace(4)* }
%struct._ZTS6vector.vector = type { i32 addrspace(4)* }

$_ZTS3foo = comdat any

$_ZN8iteratorI6vectorEC2Ev = comdat any

define weak_odr dso_local spir_kernel void @_ZTS3foo() #0 comdat !dbg !12 !kernel_arg_addr_space !10 !kernel_arg_access_qual !10 !kernel_arg_type !10 !kernel_arg_base_type !10 !kernel_arg_type_qual !10 !kernel_arg_host_accessible !10 !kernel_arg_pipe_depth !10 !kernel_arg_pipe_io !10 !kernel_arg_buffer_location !10 {
entry:
  %0 = alloca %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon", align 1
  %1 = bitcast %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon"* %0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %1) #5
  call void @llvm.dbg.declare(metadata %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon"* %0, metadata !15, metadata !DIExpression()), !dbg !23
  %2 = addrspacecast %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon"* %0 to %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)*, !dbg !24
  call spir_func void @"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEv"(%"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)* %2), !dbg !24
  %3 = bitcast %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon"* %0 to i8*, !dbg !23
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %3) #5, !dbg !23
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

; Function Attrs: inlinehint
define internal spir_func void @"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEv"(%"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)* %this) #3 align 2 !dbg !26 {
entry:
  %this.addr = alloca %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)*, align 8
  %IV = alloca %struct._ZTS8iteratorI6vectorE.iterator, align 8
  %V = alloca %struct._ZTS6vector.vector, align 8
  store %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)* %this, %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)** %this.addr, align 8, !tbaa !52
  call void @llvm.dbg.declare(metadata %"class._ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEEUlvE_.anon" addrspace(4)** %this.addr, metadata !28, metadata !DIExpression()), !dbg !56
  %0 = bitcast %struct._ZTS8iteratorI6vectorE.iterator* %IV to i8*, !dbg !57
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %0) #5, !dbg !57
  call void @llvm.dbg.declare(metadata %struct._ZTS8iteratorI6vectorE.iterator* %IV, metadata !30, metadata !DIExpression()), !dbg !58
  %1 = addrspacecast %struct._ZTS8iteratorI6vectorE.iterator* %IV to %struct._ZTS8iteratorI6vectorE.iterator addrspace(4)*, !dbg !58
  call spir_func void @_ZN8iteratorI6vectorEC2Ev(%struct._ZTS8iteratorI6vectorE.iterator addrspace(4)* %1), !dbg !58
  %2 = bitcast %struct._ZTS6vector.vector* %V to i8*, !dbg !59
  call void @llvm.lifetime.start.p0i8(i64 8, i8* %2) #5, !dbg !59
  call void @llvm.dbg.declare(metadata %struct._ZTS6vector.vector* %V, metadata !51, metadata !DIExpression()), !dbg !60
  %3 = bitcast %struct._ZTS6vector.vector* %V to i8*, !dbg !61
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %3) #5, !dbg !61
  %4 = bitcast %struct._ZTS8iteratorI6vectorE.iterator* %IV to i8*, !dbg !61
  call void @llvm.lifetime.end.p0i8(i64 8, i8* %4) #5, !dbg !61
  ret void, !dbg !61
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nounwind
define linkonce_odr dso_local spir_func void @_ZN8iteratorI6vectorEC2Ev(%struct._ZTS8iteratorI6vectorE.iterator addrspace(4)* %this) unnamed_addr #4 comdat align 2 !dbg !62 {
entry:
  %this.addr = alloca %struct._ZTS8iteratorI6vectorE.iterator addrspace(4)*, align 8
  store %struct._ZTS8iteratorI6vectorE.iterator addrspace(4)* %this, %struct._ZTS8iteratorI6vectorE.iterator addrspace(4)** %this.addr, align 8, !tbaa !52
  call void @llvm.dbg.declare(metadata %struct._ZTS8iteratorI6vectorE.iterator addrspace(4)** %this.addr, metadata !64, metadata !DIExpression()), !dbg !66
  %this1 = load %struct._ZTS8iteratorI6vectorE.iterator addrspace(4)*, %struct._ZTS8iteratorI6vectorE.iterator addrspace(4)** %this.addr, align 8
  %Itr = getelementptr inbounds %struct._ZTS8iteratorI6vectorE.iterator, %struct._ZTS8iteratorI6vectorE.iterator addrspace(4)* %this1, i32 0, i32 0, !dbg !67
  store i32 addrspace(4)* null, i32 addrspace(4)* addrspace(4)* %Itr, align 8, !dbg !67, !tbaa !68
  ret void, !dbg !70
}

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "sycl-module-id"="recursive_debug_info.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { nounwind readnone speculatable willreturn }
attributes #3 = { inlinehint "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5, !6, !7}
!opencl.spir.version = !{!8}
!spirv.Source = !{!9}
!opencl.used.extensions = !{!10}
!opencl.used.optional.core.features = !{!10}
!opencl.compiler.options = !{!10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_11, file: !1, producer: "clang version 10.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, nameTableKind: None)
!1 = !DIFile(filename: "recursive_debug_info.cpp", directory: "/localdisk/test")
!2 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!3 = !{null}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !2, size: 64)
!5 = !{i32 7, !"Dwarf Version", i32 4}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{i32 1, i32 2}
!9 = !{i32 4, i32 100000}
!10 = !{}
!11 = !{!"clang version 10.0.0"}
!12 = distinct !DISubprogram(name: "_ZTS3foo", scope: !1, file: !1, line: 28, type: !13, flags: DIFlagArtificial | DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !14)
!13 = !DISubroutineType(cc: DW_CC_LLVM_OpenCLKernel, types: !3)
!14 = !{!15}
!15 = !DILocalVariable(scope: !12, file: !1, type: !16)
!16 = distinct !DICompositeType(tag: DW_TAG_class_type, file: !1, line: 28, size: 8, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !17)
!17 = !{!18}
!18 = !DISubprogram(name: "operator()", scope: !16, file: !1, line: 28, type: !19, scopeLine: 28, flags: DIFlagPublic | DIFlagPrototyped, spFlags: DISPFlagOptimized)
!19 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !20)
!20 = !{null, !21}
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!22 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !16)
!23 = !DILocation(line: 0, scope: !12)
!24 = !DILocation(line: 0, scope: !25)
!25 = distinct !DILexicalBlock(scope: !12, file: !1)
!26 = distinct !DISubprogram(name: "operator()", linkageName: "_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlvE_clEv", scope: !16, file: !1, line: 28, type: !19, scopeLine: 28, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition | DISPFlagOptimized, unit: !0, declaration: !18, retainedNodes: !27)
!27 = !{!28, !30, !51}
!28 = !DILocalVariable(name: "this", arg: 1, scope: !26, type: !29, flags: DIFlagArtificial | DIFlagObjectPointer)
!29 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 64)
!30 = !DILocalVariable(name: "IV", scope: !26, file: !1, line: 29, type: !31)
!31 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "iterator<vector>", file: !1, line: 6, size: 64, flags: DIFlagTypePassByValue | DIFlagNonTrivial, elements: !32, templateParams: !41, identifier: "_ZTS8iteratorI6vectorE")
!32 = !{!33, !34, !38}
!33 = !DIDerivedType(tag: DW_TAG_member, name: "Itr", scope: !31, file: !1, line: 7, baseType: !4, size: 64)
!34 = !DISubprogram(name: "iterator", scope: !31, file: !1, line: 9, type: !35, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!35 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !36)
!36 = !{null, !37}
!37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !31, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!38 = !DISubprogram(name: "iterator", scope: !31, file: !1, line: 10, type: !39, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!39 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !40)
!40 = !{null, !37, !4}
!41 = !{!42}
!42 = !DITemplateTypeParameter(name: "Container", type: !43)
!43 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "vector", file: !1, line: 13, size: 64, flags: DIFlagTypePassByValue, elements: !44, identifier: "_ZTS6vector")
!44 = !{!45, !46}
!45 = !DIDerivedType(tag: DW_TAG_member, name: "Start", scope: !43, file: !1, line: 14, baseType: !4, size: 64)
!46 = !DISubprogram(name: "begin", linkageName: "_ZN6vector5beginEv", scope: !43, file: !1, line: 18, type: !47, scopeLine: 18, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized)
!47 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !48)
!48 = !{!49, !50}
!49 = !DIDerivedType(tag: DW_TAG_typedef, name: "vec_it", scope: !43, file: !1, line: 16, baseType: !31)
!50 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !43, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!51 = !DILocalVariable(name: "V", scope: !26, file: !1, line: 30, type: !43)
!52 = !{!53, !53, i64 0}
!53 = !{!"any pointer", !54, i64 0}
!54 = !{!"omnipotent char", !55, i64 0}
!55 = !{!"Simple C++ TBAA"}
!56 = !DILocation(line: 0, scope: !26)
!57 = !DILocation(line: 29, column: 7, scope: !26)
!58 = !DILocation(line: 29, column: 24, scope: !26)
!59 = !DILocation(line: 30, column: 7, scope: !26)
!60 = !DILocation(line: 30, column: 14, scope: !26)
!61 = !DILocation(line: 31, column: 5, scope: !26)
!62 = distinct !DISubprogram(name: "iterator", linkageName: "_ZN8iteratorI6vectorEC2Ev", scope: !31, file: !1, line: 9, type: !35, scopeLine: 9, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, declaration: !34, retainedNodes: !63)
!63 = !{!64}
!64 = !DILocalVariable(name: "this", arg: 1, scope: !62, type: !65, flags: DIFlagArtificial | DIFlagObjectPointer)
!65 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !31, size: 64)
!66 = !DILocation(line: 0, scope: !62)
!67 = !DILocation(line: 9, column: 16, scope: !62)
!68 = !{!69, !53, i64 0}
!69 = !{!"_ZTS8iteratorI6vectorE", !53, i64 0}
!70 = !DILocation(line: 9, column: 30, scope: !62)
