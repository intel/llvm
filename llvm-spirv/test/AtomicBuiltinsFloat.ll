; Check that translator generates atomic instructions for atomic builtins
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

; CHECK-LABEL: Label
; CHECK: Store
; CHECK-COUNT-3: AtomicStore
; CHECK-COUNT-3: AtomicLoad
; CHECK-COUNT-3: AtomicExchange

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: convergent norecurse nounwind
define dso_local spir_kernel void @test_atomic_kernel(ptr addrspace(3) %ff) local_unnamed_addr #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !4 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %0 = addrspacecast ptr addrspace(3) %ff to ptr addrspace(4)
  tail call spir_func void @_Z11atomic_initPU3AS4VU7_Atomicff(ptr addrspace(4) %0, float 1.000000e+00) #2
  tail call spir_func void @_Z12atomic_storePU3AS4VU7_Atomicff(ptr addrspace(4) %0, float 1.000000e+00) #2
  tail call spir_func void @_Z21atomic_store_explicitPU3AS4VU7_Atomicff12memory_order(ptr addrspace(4) %0, float 1.000000e+00, i32 0) #2
  tail call spir_func void @_Z21atomic_store_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(ptr addrspace(4) %0, float 1.000000e+00, i32 0, i32 1) #2
  %call = tail call spir_func float @_Z11atomic_loadPU3AS4VU7_Atomicf(ptr addrspace(4) %0) #2
  %call1 = tail call spir_func float @_Z20atomic_load_explicitPU3AS4VU7_Atomicf12memory_order(ptr addrspace(4) %0, i32 0) #2
  %call2 = tail call spir_func float @_Z20atomic_load_explicitPU3AS4VU7_Atomicf12memory_order12memory_scope(ptr addrspace(4) %0, i32 0, i32 1) #2
  %call3 = tail call spir_func float @_Z15atomic_exchangePU3AS4VU7_Atomicff(ptr addrspace(4) %0, float 1.000000e+00) #2
  %call4 = tail call spir_func float @_Z24atomic_exchange_explicitPU3AS4VU7_Atomicff12memory_order(ptr addrspace(4) %0, float 1.000000e+00, i32 0) #2
  %call5 = tail call spir_func float @_Z24atomic_exchange_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(ptr addrspace(4) %0, float 1.000000e+00, i32 0, i32 1) #2
  ret void
}

; Function Attrs: convergent
declare spir_func void @_Z11atomic_initPU3AS4VU7_Atomicff(ptr addrspace(4), float) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func void @_Z12atomic_storePU3AS4VU7_Atomicff(ptr addrspace(4), float) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func void @_Z21atomic_store_explicitPU3AS4VU7_Atomicff12memory_order(ptr addrspace(4), float, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func void @_Z21atomic_store_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(ptr addrspace(4), float, i32, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z11atomic_loadPU3AS4VU7_Atomicf(ptr addrspace(4)) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z20atomic_load_explicitPU3AS4VU7_Atomicf12memory_order(ptr addrspace(4), i32) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z20atomic_load_explicitPU3AS4VU7_Atomicf12memory_order12memory_scope(ptr addrspace(4), i32, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z15atomic_exchangePU3AS4VU7_Atomicff(ptr addrspace(4), float) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z24atomic_exchange_explicitPU3AS4VU7_Atomicff12memory_order(ptr addrspace(4), float, i32) local_unnamed_addr #1

; Function Attrs: convergent
declare spir_func float @_Z24atomic_exchange_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(ptr addrspace(4), float, i32, i32) local_unnamed_addr #1

attributes #0 = { convergent norecurse nounwind "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" }
attributes #1 = { convergent "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent nounwind }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{i32 3}
!3 = !{!"none"}
!4 = !{!"atomic_float*"}
!5 = !{!"_Atomic(float)*"}
!6 = !{!"volatile"}
