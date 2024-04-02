; Check that translator doesn't crash when translating debug info for
; __translate_sampler_initializer() call with a constant argument;
; LLVMToSPIRVDbgTran::transLocationInfo() method assumes that LLVM instructions
; in the basic block are mapped to SPIR-V instructions in the corresponding
; basic block. That's not true for __translate_sampler_initializer() call with
; constant argument. Such call is being translated to an OpConstantSampler in
; global scope.

; Original .cl source:
; void foo() {
;   const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
; }

; Command line:
; clang -cc1 -triple spir constant_sampler.cl -cl-std=cl2.0 -emit-llvm -o llvm-spirv/test/DebugInfo/translate_sampler_initializer.ll -finclude-default-header -debug-info-kind=standalone

; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s

; CHECK: TypeSampler [[#SamplerTy:]]
; CHECK: ConstantSampler [[#SamplerTy]] [[#ConstSampler:]] 2 0 0

; CHECK: ExtInst [[#]] [[#Func:]] [[#]] DebugFunction
; CHECK: ExtInst [[#]] [[#SamplerVar:]] [[#]] DebugLocalVariable

; CHECK: Label
; CHECK-NEXT: ExtInst [[#]] [[#]] [[#]] DebugScope [[#Func]] 
; CHECK-NEXT: Line [[#]] 0 0
; CHECK-NEXT: ExtInst [[#]] [[#]] [[#]] DebugValue [[#SamplerVar]] [[#ConstSampler]]
; CHECK-NEXT: Line [[#]] 4 0
; CHECK-NEXT: Return

source_filename = "constant_sampler.cl"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

%opencl.sampler_t = type opaque

; Function Attrs: norecurse nounwind
define spir_func void @foo() local_unnamed_addr #0 !dbg !7 {
entry:
  %0 = tail call ptr addrspace(2) @__translate_sampler_initializer(i32 20) #2, !dbg !17
  call void @llvm.dbg.value(metadata ptr addrspace(2) %0, metadata !12, metadata !DIExpression()), !dbg !18
  ret void, !dbg !19
}

declare ptr addrspace(2) @__translate_sampler_initializer(i32) local_unnamed_addr

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable willreturn }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!opencl.ocl.version = !{!5}
!opencl.spir.version = !{!5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0 (https://github.com/llvm/llvm-project.git 21caba599e6ce806abc492b7ed1653a1aed8b63c)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 2, i32 0}
!6 = !{!"clang version 11.0.0 (https://github.com/llvm/llvm-project.git 21caba599e6ce806abc492b7ed1653a1aed8b63c)"}
!7 = distinct !DISubprogram(name: "foo", scope: !8, file: !8, line: 1, type: !9, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !11)
!8 = !DIFile(filename: "constant_sampler.cl", directory: "/tmp")
!9 = !DISubroutineType(cc: DW_CC_LLVM_SpirFunction, types: !10)
!10 = !{null}
!11 = !{!12}
!12 = !DILocalVariable(name: "sampler", scope: !7, file: !8, line: 3, type: !13)
!13 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !14)
!14 = !DIDerivedType(tag: DW_TAG_typedef, name: "sampler_t", file: !1, line: 3, baseType: !15)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 32)
!16 = !DICompositeType(tag: DW_TAG_structure_type, name: "opencl_sampler_t", file: !1, flags: DIFlagFwdDecl)
!17 = !DILocation(line: 3, scope: !7)
!18 = !DILocation(line: 0, scope: !7)
!19 = !DILocation(line: 4, scope: !7)
