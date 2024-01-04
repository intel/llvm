; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv --spirv-debug-info-version=ocl-100 %t.bc
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck %s --input-file %t.spt -check-prefixes=CHECK-SPIRV,CHECK-SPIRV-NO-BITFIELD
; RUN: llvm-spirv -r %t.spv -o %t_1.bc
; RUN: llvm-dis %t_1.bc -o %t_1.ll
; RUN: FileCheck %s --input-file %t_1.ll  -check-prefixes=CHECK-LLVM,CHECK-LLVM-NO-BITFIELD

; RUN: llvm-spirv --spirv-ext=+SPV_KHR_non_semantic_info --spirv-debug-info-version=nonsemantic-shader-100 %t.bc
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck %s --input-file %t.spt -check-prefixes=CHECK-SPIRV,CHECK-SPIRV-BITFIELD
; RUN: llvm-spirv -r %t.spv -o %t_1.bc
; RUN: llvm-dis %t_1.bc -o %t_1.ll
; RUN: FileCheck %s --input-file %t_1.ll  -check-prefixes=CHECK-LLVM,CHECK-LLVM-NO-BITFIELD

; RUN: llvm-spirv --spirv-debug-info-version=nonsemantic-shader-200 %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck %s --input-file %t.spt -check-prefixes=CHECK-SPIRV,CHECK-SPIRV-BITFIELD
; RUN: llvm-spirv -r %t.spv -o %t_1.bc
; RUN: llvm-dis %t_1.bc -o %t_1.ll
; RUN: FileCheck %s --input-file %t_1.ll  -check-prefixes=CHECK-LLVM,CHECK-LLVM-BITFIELD

; CHECK-SPIRV: String [[#A:]] "a"
; CHECK-SPIRV: String [[#B:]] "b"
; CHECK-SPIRV: String [[#C:]] "c"
; CHECK-SPIRV: String [[#D:]] "d"
; CHECK-SPIRV: String [[#E:]] "e"
; CHECK-SPIRV: String [[#F:]] "f"

; CHECK-SPIRV-DAG: Constant [[#]] [[#THREE:]] 3 {{$}}
; CHECK-SPIRV-DAG: Constant [[#]] [[#ONE:]] 1 {{$}}

; CHECK-SPIRV-NO-BITFIELD: ExtInst [[#]] [[#]] [[#]] DebugTypeMember [[#A]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]]
; CHECK-SPIRV-NO-BITFIELD: ExtInst [[#]] [[#]] [[#]] DebugTypeMember [[#B]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]]
; CHECK-SPIRV-NO-BITFIELD: ExtInst [[#]] [[#]] [[#]] DebugTypeMember [[#C]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]]
; CHECK-SPIRV-NO-BITFIELD: ExtInst [[#]] [[#]] [[#]] DebugTypeMember [[#D]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]]
; CHECK-SPIRV-NO-BITFIELD: ExtInst [[#]] [[#]] [[#]] DebugTypeMember [[#E]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]]
; CHECK-SPIRV-NO-BITFIELD: ExtInst [[#]] [[#]] [[#]] DebugTypeMember [[#F]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]]

; CHECK-SPIRV-BITFIELD: ExtInst [[#]] [[#]] [[#]] DebugTypeMember [[#A]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#THREE]] [[#]]
; CHECK-SPIRV-BITFIELD: ExtInst [[#]] [[#]] [[#]] DebugTypeMember [[#B]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#THREE]] [[#]]
; CHECK-SPIRV-BITFIELD: ExtInst [[#]] [[#]] [[#]] DebugTypeMember [[#C]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#THREE]] [[#]]
; CHECK-SPIRV-BITFIELD: ExtInst [[#]] [[#]] [[#]] DebugTypeMember [[#D]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#THREE]] [[#]]
; CHECK-SPIRV-BITFIELD: ExtInst [[#]] [[#]] [[#]] DebugTypeMember [[#E]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#THREE]] [[#]]
; CHECK-SPIRV-BITFIELD: ExtInst [[#]] [[#]] [[#]] DebugTypeMember [[#F]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#ONE]] [[#]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%struct.struct_bit_fields1 = type { i16 }

@__const._Z3fooii.arr_bf1 = private unnamed_addr addrspace(1) constant [1 x { i8, i8 }] [{ i8, i8 } { i8 -47, i8 8 }], align 2

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define dso_local spir_func noundef i32 @_Z3fooii(i32 noundef %0, i32 noundef %1) #0 !dbg !10 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca [1 x %struct.struct_bit_fields1], align 2
  %7 = alloca i32, align 4
  %8 = addrspacecast ptr %3 to ptr addrspace(4)
  %9 = addrspacecast ptr %4 to ptr addrspace(4)
  %10 = addrspacecast ptr %5 to ptr addrspace(4)
  %11 = addrspacecast ptr %6 to ptr addrspace(4)
  %12 = addrspacecast ptr %7 to ptr addrspace(4)
  store i32 %0, ptr addrspace(4) %9, align 4
  call void @llvm.dbg.declare(metadata ptr %4, metadata !15, metadata !DIExpression()), !dbg !16
  store i32 %1, ptr addrspace(4) %10, align 4
  call void @llvm.dbg.declare(metadata ptr %5, metadata !17, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata ptr %6, metadata !19, metadata !DIExpression()), !dbg !32
  call void @llvm.memcpy.p4.p1.i64(ptr addrspace(4) align 2 %11, ptr addrspace(1) align 2 @__const._Z3fooii.arr_bf1, i64 2, i1 false), !dbg !32
  call void @llvm.dbg.declare(metadata ptr %7, metadata !33, metadata !DIExpression()), !dbg !34
  store i32 0, ptr addrspace(4) %12, align 4, !dbg !34
  %13 = load i32, ptr addrspace(4) %12, align 4, !dbg !35
  ret i32 %13, !dbg !36
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p4.p1.i64(ptr addrspace(4) noalias nocapture writeonly, ptr addrspace(1) noalias nocapture readonly, i64, i1 immarg) #2

attributes #0 = { convergent mustprogress noinline norecurse nounwind optnone "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="bitfields-packed.cpp" "unsafe-fp-math"="true" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.dbg.cu = !{!0}
!opencl.spir.version = !{!2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2, !2}
!spirv.Source = !{!3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3, !3}
!opencl.compiler.options = !{!4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4, !4}
!llvm.ident = !{!5, !5, !5, !5, !5, !5, !5, !5, !5, !5, !5, !5, !5, !5, !5, !5, !5, !5, !5}
!llvm.module.flags = !{!6, !7, !8, !9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: false, flags: "", runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "bitfields-packed.cpp", directory: "/tmp")
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{}
!5 = !{!"Compiler"}
!6 = !{i32 7, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 7, !"frame-pointer", i32 2}
!10 = distinct !DISubprogram(name: "foo", linkageName: "_Z3fooii", scope: !11, file: !11, line: 29, type: !12, scopeLine: 29, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !4)
!11 = !DIFile(filename: "bitfields-packed.cpp", directory: "/tmp")
!12 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !13)
!13 = !{!14, !14, !14}
!14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!15 = !DILocalVariable(name: "a", arg: 1, scope: !10, file: !11, line: 29, type: !14)
!16 = !DILocation(line: 29, column: 43, scope: !10)
!17 = !DILocalVariable(name: "b", arg: 2, scope: !10, file: !11, line: 29, type: !14)
!18 = !DILocation(line: 29, column: 50, scope: !10)
; CHECK-LLVM: !DILocalVariable(name: "arr_bf1", {{.*}}, type: ![[#ARRAY_TYPE:]])
!19 = !DILocalVariable(name: "arr_bf1", scope: !10, file: !11, line: 33, type: !20)
; CHECK-LLVM: ![[#ARRAY_TYPE]] = {{.*}}!DICompositeType(tag: DW_TAG_array_type, baseType: ![[#BITFIELD_STRUCT:]], {{.*}}) 
!20 = !DICompositeType(tag: DW_TAG_array_type, baseType: !21, size: 16, elements: !30)
; CHECK-LLVM: ![[#BITFIELD_STRUCT]] = {{.*}}!DICompositeType(tag: DW_TAG_structure_type, name: "struct_bit_fields1", {{.*}}, identifier: "_ZTS18struct_bit_fields1")
!21 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "struct_bit_fields1", file: !11, line: 19, size: 16, flags: DIFlagTypePassByValue, elements: !22, identifier: "_ZTS18struct_bit_fields1")
!22 = !{!23, !25, !26, !27, !28, !29}
; CHECK-LLVM-BITFIELD: !DIDerivedType(tag: DW_TAG_member, name: "a", scope: ![[#BITFIELD_STRUCT]], {{.*}}, size: 3, flags: {{.*}}DIFlagBitField{{.*}})
; CHECK-LLVM-BITFIELD: !DIDerivedType(tag: DW_TAG_member, name: "b", scope: ![[#BITFIELD_STRUCT]], {{.*}}, size: 3, offset: 3, flags: {{.*}}DIFlagBitField{{.*}})
; CHECK-LLVM-BITFIELD: !DIDerivedType(tag: DW_TAG_member, name: "c", scope: ![[#BITFIELD_STRUCT]], {{.*}}, size: 3, offset: 6, flags: {{.*}}DIFlagBitField{{.*}})
; CHECK-LLVM-BITFIELD: !DIDerivedType(tag: DW_TAG_member, name: "d", scope: ![[#BITFIELD_STRUCT]], {{.*}}, size: 3, offset: 9, flags: {{.*}}DIFlagBitField{{.*}})
; CHECK-LLVM-BITFIELD: !DIDerivedType(tag: DW_TAG_member, name: "e", scope: ![[#BITFIELD_STRUCT]], {{.*}}, size: 3, offset: 12, flags: {{.*}}DIFlagBitField{{.*}})
; CHECK-LLVM-BITFIELD: !DIDerivedType(tag: DW_TAG_member, name: "f", scope: ![[#BITFIELD_STRUCT]], {{.*}}, size: 1, offset: 15, flags: {{.*}}DIFlagBitField{{.*}})
; CHECK-LLVM-NO-BITFIELD-NOT: DIFlagBitField
!23 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !21, file: !11, line: 21, baseType: !24, size: 3, flags: DIFlagBitField, extraData: i64 0)
!24 = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
!25 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !21, file: !11, line: 22, baseType: !24, size: 3, offset: 3, flags: DIFlagBitField, extraData: i64 0)
!26 = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: !21, file: !11, line: 23, baseType: !24, size: 3, offset: 6, flags: DIFlagBitField, extraData: i64 0)
!27 = !DIDerivedType(tag: DW_TAG_member, name: "d", scope: !21, file: !11, line: 24, baseType: !24, size: 3, offset: 9, flags: DIFlagBitField, extraData: i64 0)
!28 = !DIDerivedType(tag: DW_TAG_member, name: "e", scope: !21, file: !11, line: 25, baseType: !24, size: 3, offset: 12, flags: DIFlagBitField, extraData: i64 0)
!29 = !DIDerivedType(tag: DW_TAG_member, name: "f", scope: !21, file: !11, line: 26, baseType: !24, size: 1, offset: 15, flags: DIFlagBitField, extraData: i64 0)
!30 = !{!31}
!31 = !DISubrange(count: 1)
!32 = !DILocation(line: 33, column: 22, scope: !10)
!33 = !DILocalVariable(name: "ans", scope: !10, file: !11, line: 35, type: !14)
!34 = !DILocation(line: 35, column: 7, scope: !10)
!35 = !DILocation(line: 36, column: 10, scope: !10)
!36 = !DILocation(line: 36, column: 3, scope: !10)
