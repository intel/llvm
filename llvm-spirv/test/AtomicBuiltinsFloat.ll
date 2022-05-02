; Check that translator generates atomic instructions for atomic builtins
; FP-typed atomic_fetch_sub and atomic_fetch_sub_explicit should be translated
; to FunctionCall
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

; CHECK-LABEL: Label
; CHECK: Store
; CHECK-COUNT-3: AtomicStore
; CHECK-COUNT-3: AtomicLoad
; CHECK-COUNT-3: AtomicExchange
; CHECK-COUNT-3: FunctionCall

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: convergent norecurse nounwind
define dso_local spir_kernel void @test_atomic_kernel(float addrspace(3)* %ff) local_unnamed_addr #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %0 = addrspacecast float addrspace(3)* %ff to float addrspace(4)*
  tail call spir_func void @_Z11atomic_initPU3AS4VU7_Atomicff(float addrspace(4)* %0, float 1.000000e+00) #2
  tail call spir_func void @_Z12atomic_storePU3AS4VU7_Atomicff(float addrspace(4)* %0, float 1.000000e+00) #2
  tail call spir_func void @_Z21atomic_store_explicitPU3AS4VU7_Atomicff12memory_order(float addrspace(4)* %0, float 1.000000e+00, i32 0) #2
  tail call spir_func void @_Z21atomic_store_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(float addrspace(4)* %0, float 1.000000e+00, i32 0, i32 1) #2
  %call = tail call spir_func float @_Z11atomic_loadPU3AS4VU7_Atomicf(float addrspace(4)* %0) #2
  %call1 = tail call spir_func float @_Z20atomic_load_explicitPU3AS4VU7_Atomicf12memory_order(float addrspace(4)* %0, i32 0) #2
  %call2 = tail call spir_func float @_Z20atomic_load_explicitPU3AS4VU7_Atomicf12memory_order12memory_scope(float addrspace(4)* %0, i32 0, i32 1) #2
  %call3 = tail call spir_func float @_Z15atomic_exchangePU3AS4VU7_Atomicff(float addrspace(4)* %0, float 1.000000e+00) #2
  %call4 = tail call spir_func float @_Z24atomic_exchange_explicitPU3AS4VU7_Atomicff12memory_order(float addrspace(4)* %0, float 1.000000e+00, i32 0) #2
  %call5 = tail call spir_func float @_Z24atomic_exchange_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(float addrspace(4)* %0, float 1.000000e+00, i32 0, i32 1) #2
  %call6 = tail call spir_func float @_Z16atomic_fetch_subPU3AS3VU7_Atomicff(float addrspace(3)* %ff, float 1.000000e+00) #2
  %call7 = tail call spir_func float @_Z25atomic_fetch_sub_explicitPU3AS3VU7_Atomicff12memory_order(float addrspace(3)* %ff, float 1.000000e+00, i32 0) #2
  %call8 = tail call spir_func float @_Z25atomic_fetch_sub_explicitPU3AS3VU7_Atomicff12memory_order12memory_scope(float addrspace(3)* %ff, float 1.000000e+00, i32 0, i32 1) #2
  ret void
}

; Function Attrs: convergent
declare spir_func void @_Z11atomic_initPU3AS4VU7_Atomicff(float addrspace(4)*, float) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func void @_Z12atomic_storePU3AS4VU7_Atomicff(float addrspace(4)*, float) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func void @_Z21atomic_store_explicitPU3AS4VU7_Atomicff12memory_order(float addrspace(4)*, float, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func void @_Z21atomic_store_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(float addrspace(4)*, float, i32, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z11atomic_loadPU3AS4VU7_Atomicf(float addrspace(4)*) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z20atomic_load_explicitPU3AS4VU7_Atomicf12memory_order(float addrspace(4)*, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z20atomic_load_explicitPU3AS4VU7_Atomicf12memory_order12memory_scope(float addrspace(4)*, i32, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z15atomic_exchangePU3AS4VU7_Atomicff(float addrspace(4)*, float) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z24atomic_exchange_explicitPU3AS4VU7_Atomicff12memory_order(float addrspace(4)*, float, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z24atomic_exchange_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(float addrspace(4)*, float, i32, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z16atomic_fetch_subPU3AS3VU7_Atomicff(float addrspace(3)*, float) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z25atomic_fetch_sub_explicitPU3AS3VU7_Atomicff12memory_order(float addrspace(3)*, float, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z25atomic_fetch_sub_explicitPU3AS3VU7_Atomicff12memory_order12memory_scope(float addrspace(3)*, float, i32, i32) local_unnamed_addr #1

attributes #0 = { convergent norecurse nounwind "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" }
attributes #1 = { convergent "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 14.0.0 (https://github.com/llvm/llvm-project.git 28c4f97a1dc8608cdd4db452b73d7d4afc89acc9)"}
!3 = !{i32 3}
!4 = !{!"none"}
!5 = !{!"atomic_float*"}
!6 = !{!"_Atomic(float)*"}
!7 = !{!"volatile"}
