; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_EXT_shader_atomic_float_add -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s -check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -r --spirv-target-env=CL2.0 -o - | llvm-dis -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s -check-prefix=CHECK-LLVM

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

@j = local_unnamed_addr addrspace(1) global i32 0, align 4

; SPIRV scopes
; 0 - ScopeCrossDevice (all svm devices)
; 1 - ScopeDevice
; 2 - ScopeWorkGroup
; 3 - ScopeSubgroup
; 4 - ScopeInvocation (mapped from work item)

; OCL scopes
; 0 - work_item
; 1 - work_group
; 2 - device
; 3 - all_svm_devices
; 4 - sub_group

; CHECK-SPIRV-DAG: Constant [[#]] [[#ConstInt0:]] 0
; CHECK-SPIRV-DAG: Constant [[#]] [[#SequentiallyConsistent:]] 16
; CHECK-SPIRV-DAG: Constant [[#]] [[#ConstInt1:]] 1
; CHECK-SPIRV-DAG: Constant [[#]] [[#ConstInt2:]] 2
; CHECK-SPIRV-DAG: Constant [[#]] [[#ConstInt3:]] 3
; CHECK-SPIRV-DAG: Constant [[#]] [[#ConstInt4:]] 4
; CHECK-SPIRV-DAG: Constant [[#]] [[#Const2Power30:]] 1073741824
; CHECK-SPIRV-DAG: Constant [[#]] [[#ConstInt42:]] 42

; AtomicLoad ResTypeId ResId PtrId MemScopeId MemSemanticsId
; CHECK-SPIRV: AtomicLoad [[#]] [[#]] [[#]] [[#ConstInt2]] [[#SequentiallyConsistent]]
; CHECK-SPIRV: AtomicLoad [[#]] [[#]] [[#]] [[#ConstInt1]] [[#SequentiallyConsistent]]
; CHECK-SPIRV: AtomicLoad [[#]] [[#]] [[#]] [[#ConstInt1]] [[#SequentiallyConsistent]]
; CHECK-SPIRV: AtomicLoad [[#]] [[#]] [[#]] [[#ConstInt3]] [[#SequentiallyConsistent]]

; CHECK-LLVM: call spir_func i32 @_Z20atomic_load_explicitPU3AS4VU7_Atomici12memory_order12memory_scope(ptr{{.*}}, i32 5, i32 1)
; CHECK-LLVM: call spir_func i32 @_Z20atomic_load_explicitPU3AS4VU7_Atomici12memory_order12memory_scope(ptr{{.*}}, i32 5, i32 2)
; CHECK-LLVM: call spir_func i32 @_Z20atomic_load_explicitPU3AS4VU7_Atomici12memory_order12memory_scope(ptr{{.*}}, i32 5, i32 2)
; CHECK-LLVM: call spir_func i32 @_Z20atomic_load_explicitPU3AS4VU7_Atomici12memory_order12memory_scope(ptr{{.*}}, i32 5, i32 4)

define dso_local void @fi1(ptr addrspace(4) nocapture noundef readonly %i) local_unnamed_addr #0 {
entry:
  %0 = load atomic i32, ptr addrspace(4) %i syncscope("workgroup") seq_cst, align 4
  %1 = load atomic i32, ptr addrspace(4) %i syncscope("device") seq_cst, align 4
  %2 = load atomic i32, ptr addrspace(4) %i seq_cst, align 4
  %3 = load atomic i32, ptr addrspace(4) %i syncscope("subgroup") seq_cst, align 4
  ret void
}

; AtomicStore PtrId MemScopeId MemSemanticsId ValueId
; CHECK-SPIRV: AtomicStore [[#]] [[#ConstInt3]] [[#SequentiallyConsistent]] [[#ConstInt1]]
; CHECK-SPIRV: AtomicStore [[#]] [[#ConstInt2]] [[#SequentiallyConsistent]] [[#ConstInt1]]
; CHECK-LLVM: call spir_func void @_Z21atomic_store_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(ptr{{.*}}, i32 5, i32 4)
; CHECK-LLVM: call spir_func void @_Z21atomic_store_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(ptr{{.*}}, i32 5, i32 1)

define dso_local void @test_addr(ptr addrspace(1) nocapture noundef writeonly %ig, ptr addrspace(3) nocapture noundef writeonly %il) local_unnamed_addr #0 {
entry:
  store atomic i32 1, ptr addrspace(1) %ig syncscope("subgroup") seq_cst, align 4
  store atomic i32 1, ptr addrspace(3) %il syncscope("workgroup") seq_cst, align 4
  ret void
}

; Atomic* ResTypeId ResId PtrId MemScopeId MemSemanticsId ValueId
; CHECK-SPIRV: AtomicAnd [[#]] [[#]] [[#]] [[#ConstInt4]] [[#SequentiallyConsistent]] [[#ConstInt1]]
; CHECK-SPIRV: AtomicSMin [[#]] [[#]] [[#]] [[#ConstInt0]] [[#SequentiallyConsistent]] [[#ConstInt1]]
; CHECK-SPIRV: AtomicSMax [[#]] [[#]] [[#]] [[#ConstInt1]] [[#SequentiallyConsistent]] [[#ConstInt1]]
; CHECK-SPIRV: AtomicUMin [[#]] [[#]] [[#]] [[#ConstInt2]] [[#SequentiallyConsistent]] [[#ConstInt1]]
; CHECK-SPIRV: AtomicUMax [[#]] [[#]] [[#]] [[#ConstInt2]] [[#SequentiallyConsistent]] [[#ConstInt1]]

; CHECK-LLVM: call spir_func i32 @_Z25atomic_fetch_and_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(ptr{{.*}}, i32 1, i32 5, i32 0)
; CHECK-LLVM: call spir_func i32 @_Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(ptr{{.*}}, i32 1, i32 5, i32 3)
; CHECK-LLVM: call spir_func i32 @_Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(ptr{{.*}}, i32 1, i32 5, i32 2)
; CHECK-LLVM: call spir_func i32 @_Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicjj12memory_order12memory_scope(ptr{{.*}}, i32 1, i32 5, i32 1)
; CHECK-LLVM: call spir_func i32 @_Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicjj12memory_order12memory_scope(ptr{{.*}}, i32 1, i32 5, i32 1)

define dso_local void @fi3(ptr nocapture noundef %i, ptr nocapture noundef %ui) local_unnamed_addr #0 {
entry:
  %0 = atomicrmw and ptr %i, i32 1 syncscope("work_item") seq_cst, align 4
  %1 = atomicrmw min ptr %i, i32 1 syncscope("all_svm_devices") seq_cst, align 4
  %2 = atomicrmw max ptr %i, i32 1 syncscope("wrong_scope") seq_cst, align 4
  %3 = atomicrmw umin ptr %ui, i32 1 syncscope("workgroup") seq_cst, align 4
  %4 = atomicrmw umax ptr %ui, i32 1 syncscope("workgroup") seq_cst, align 4
  ret void
}

; AtomicCompareExchange ResTypeId ResId PtrId MemScopeId MemSemEqualId MemSemUnequalId ValueId ComparatorId
; CHECK-SPIRV: AtomicCompareExchange [[#]] [[#]] [[#]] [[#ConstInt2]] [[#ConstInt2]] [[#ConstInt2]] [[#ConstInt1]] [[#ConstInt0]]
; CHECK-LLVM: call spir_func i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(ptr{{.*}}, ptr{{.*}}, i32 1, i32 2, i32 2, i32 1)

define dso_local zeroext i1 @fi4(ptr nocapture noundef %i) local_unnamed_addr #0 {
entry:
  %0 = cmpxchg ptr %i, i32 0, i32 1 syncscope("workgroup") acquire acquire, align 4
  %1 = extractvalue { i32, i1 } %0, 1
  ret i1 %1
}

; AtomicExchange ResTypeId ResId PtrId MemScopeId MemSemanticsId ValueId
; CHECK-SPIRV: AtomicExchange [[#]] [[#]] [[#]] [[#ConstInt2]] [[#SequentiallyConsistent]] [[#Const2Power30]]
; CHECK-LLVM: call spir_func i32 @_Z24atomic_exchange_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(ptr{{.*}}, i32 1073741824, i32 5, i32 1)

define dso_local float @ff3(ptr nocapture noundef %d) local_unnamed_addr #0 {
entry:
  %0 = atomicrmw xchg ptr %d, i32 1073741824 syncscope("workgroup") seq_cst, align 4
  %1 = bitcast i32 %0 to float
  ret float %1
}

; AtomicFAddEXT ResTypeId ResId PtrId MemScopeId MemSemanticsId ValueId
; CHECK-SPIRV: AtomicFAddEXT [[#]] [[#]] [[#]] [[#ConstInt2]] [[#ConstInt0]] [[#]]
; CHECK-LLVM: call spir_func float @_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(ptr{{.*}}, i32 0, i32 1)

define dso_local float @ff4(ptr addrspace(1) nocapture noundef %d, float noundef %a) local_unnamed_addr #0 {
entry:
  %0 = atomicrmw fadd ptr addrspace(1) %d, float %a syncscope("workgroup") monotonic, align 4
  ret float %0
}

; ; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none)
; define dso_local void @atomic_init_foo() local_unnamed_addr #1 {
; entry:
;   store i32 42, ptr addrspace(1) @j, align 4
;   ret void
; }

; Store PtrId ObjId MemOps+
; CHECK-SPIRV: Store [[#]] [[#ConstInt42]]
; CHECK-LLVM: store i32 42, ptr addrspace(1) @j

define dso_local void @atomic_init_foo() local_unnamed_addr #1 {
entry:
  tail call spir_func void @_Z11atomic_initPU3AS4VU7_Atomicff(ptr addrspace(1) @j, i32 42)
  ret void
}

; Function Attrs: convergent
declare spir_func void @_Z11atomic_initPU3AS4VU7_Atomicff(ptr addrspace(1), i32) local_unnamed_addr

attributes #0 = { mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite) "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!1}
!opencl.ocl.version = !{!2}
!llvm.ident = !{!3}

!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 2, i32 0}
!3 = !{!"clang version 18.0.0 (https://github.com/llvm/llvm-project.git 7ce613fc77af092dd6e9db71ce3747b75bc5616e)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
