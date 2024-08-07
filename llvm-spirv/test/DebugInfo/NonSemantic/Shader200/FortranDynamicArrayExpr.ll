;; The test checks, that Fortran dynamic arrays are being correctly represented
;; by SPIR-V debug information
;; Unlike 'static' arrays dynamic can have following parameters of
;; DICompositeType metadata with DW_TAG_array_type tag:
;; Data Location, Associated, Allocated and Rank which can be represented
;; by either DIExpression or DIVariable (both local and global).
;; This test if for expression representation.
;; FortranDynamicArrayVar.ll is for variable representation.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text --spirv-debug-info-version=nonsemantic-shader-200 -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-debug-info-version=nonsemantic-shader-200
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: ExtInstImport [[#Import:]] "NonSemantic.Shader.DebugInfo.200"
; CHECK-SPIRV-DAG: String [[#BasicTName:]] "INTEGER*4"
; CHECK-SPIRV-DAG: TypeInt [[#Int32T:]] 32 0
; CHECK-SPIRV-DAG: Constant [[#Int32T]] [[#IntConst:]] 32
; CHECK-SPIRV-DAG: Constant [[#Int32T]] [[#Flag:]] 4
; CHECK-SPIRV-DAG: TypeVoid [[#VoidT:]]
; CHECK-SPIRV: ExtInst [[#VoidT]] [[#DbgInfoNone:]] [[#Import]] DebugInfoNone
; CHECK-SPIRV: ExtInst [[#VoidT]] [[#ArrayBasicT:]] [[#Import]] DebugTypeBasic [[#BasicTName]] [[#IntConst]] [[#Flag]]
; CHECK-SPIRV: ExtInst [[#VoidT]] [[#DbgExprLocation:]] [[#Import]] DebugExpression [[#]] [[#]] {{$}}
; CHECK-SPIRV: ExtInst [[#VoidT]] [[#DbgExprAssociated:]] [[#Import]] DebugExpression [[#]] [[#]] [[#]] [[#]] {{$}}
; CHECK-SPIRV: ExtInst [[#VoidT]] [[#DbgExprLowerBound:]] [[#Import]] DebugExpression [[#]] [[#]] [[#]] {{$}}
; CHECK-SPIRV: ExtInst [[#VoidT]] [[#DbgExprUpperBound:]] [[#Import]] DebugExpression [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] [[#]] {{$}}
; CHECK-SPIRV: ExtInst [[#VoidT]] [[#DbgExprStride:]] [[#Import]] DebugExpression [[#]] [[#]] [[#]] {{$}}
; CHECK-SPIRV: ExtInst [[#VoidT]] [[#DbgSubRangeId:]] [[#Import]] DebugTypeSubrange [[#DbgExprLowerBound]] [[#DbgExprUpperBound]] [[#DbgInfoNone]] [[#DbgExprStride]]
; CHECK-SPIRV: ExtInst [[#VoidT]] [[#DbgArrayId:]] [[#Import]] DebugTypeArrayDynamic [[#ArrayBasicT]] [[#DbgExprLocation]] [[#DbgExprAssociated]] [[#DbgInfoNone]] [[#DbgInfoNone]] [[#DbgSubRangeId]]

; CHECK-LLVM: %[[#Array:]] = alloca
; CHECK-LLVM: #dbg_value(ptr %[[#Array]], ![[#DbgLVar:]]
; CHECK-LLVM: ![[#DbgLVar]] = !DILocalVariable(name: "pint", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[#DbgLVarT:]])
; CHECK-LLVM: ![[#DbgLVarT]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[#DbgArrayT:]], size: 64)
; CHECK-LLVM: ![[#DbgArrayT]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[#DbgArrayBaseT:]], size: 32, elements: ![[#Elements:]], dataLocation: !DIExpression(DW_OP_push_object_address, DW_OP_deref), associated: !DIExpression(DW_OP_push_object_address, DW_OP_deref, DW_OP_constu, 0, DW_OP_or))
; CHECK-LLVM: ![[#DbgArrayBaseT]] = !DIBasicType(name: "INTEGER*4", size: 32, encoding: DW_ATE_signed)
; CHECK-LLVM: ![[#Elements]] = !{![[#SubRange:]]}
; CHECK-LLVM: ![[#SubRange]] = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 64, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 64, DW_OP_deref, DW_OP_push_object_address, DW_OP_plus_uconst, 48, DW_OP_deref, DW_OP_plus, DW_OP_constu, 1, DW_OP_minus), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 56, DW_OP_deref))


; ModuleID = 'reproducer.ll'
source_filename = "test.f90"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

%qnca = type { ptr addrspace(4), i64, i64, i64, i64, i64, [1 x { i64, i64, i64 }] }

; Function Attrs: noinline nounwind optnone
define weak dso_local spir_kernel void @TEST() #0 !dbg !5 {
newFuncRoot:
  %0 = alloca %qnca, align 8
  call void @llvm.dbg.value(metadata ptr %0, metadata !8, metadata !DIExpression()), !dbg !14
  ret void
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { noinline nounwind optnone }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}
!spirv.Source = !{!4}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 2, !"Dwarf Version", i32 4}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran95, file: !3, producer: "fortran", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!3 = !DIFile(filename: "test.f90", directory: "/path/to")
!4 = !{i32 4, i32 200000}
!5 = distinct !DISubprogram(name: "test", linkageName: "MAIN__", scope: !3, file: !3, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "pint", scope: !5, file: !3, line: 3, type: !9)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64)
!10 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, elements: !12, dataLocation: !DIExpression(DW_OP_push_object_address, DW_OP_deref), associated: !DIExpression(DW_OP_push_object_address, DW_OP_deref, DW_OP_constu, 0, DW_OP_or))
!11 = !DIBasicType(name: "INTEGER*4", size: 32, encoding: DW_ATE_signed)
!12 = !{!13}
!13 = !DISubrange(lowerBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 64, DW_OP_deref), upperBound: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 64, DW_OP_deref, DW_OP_push_object_address, DW_OP_plus_uconst, 48, DW_OP_deref, DW_OP_plus, DW_OP_constu, 1, DW_OP_minus), stride: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 56, DW_OP_deref))
!14 = !DILocation(line: 1, scope: !5)

