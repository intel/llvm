; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefixes=CHECK
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64"

; CHECK-DAG: 4 TypeInt [[#int:]] 32 0
; CHECK-DAG: Constant [[#int]] [[#DeviceScope:]] 4
; CHECK-DAG: Constant [[#int]] [[#SequentiallyConsistent_MS:]] 0
; CHECK-DAG: 4 TypePointer [[#int_ptr:]] 4 [[#int]]
; CHECK-DAG: 2 TypeBool [[#bool:]]

define spir_kernel void @test_atomic_kernel() {
entry:
  %arrayidx7 = getelementptr i32, ptr addrspace(3) null, i64 0

; CHECK: PtrAccessChain [[#int_ptr]] [[#Pointer:]] [[#]] [[#]] 
; CHECK: Load [[#int]] [[#Comparator:]] [[#]]
; CHECK-NEXT: AtomicCompareExchange [[#int]] [[#Result:]] [[#Pointer]] [[#DeviceScope]] [[#SequentiallyConsistent_MS]] [[#SequentiallyConsistent_MS]] [[#]] [[#Comparator]]
; CHECK-NEXT: Store [[#]] [[#Result]]
; CHECK-NEXT: IEqual [[#bool]] [[#]] [[#Result]] [[#Comparator]]
  %call10 = call spir_func i1 @_Z37atomic_compare_exchange_weak_explicitPU3AS3VU7_AtomiciPii12memory_orderS4_12memory_scope(ptr addrspace(3) %arrayidx7, ptr null, i32 14, i32 0, i32 0, i32 0)

; CHECK: Load [[#int]] [[#Comparator:]] [[#]]
; CHECK-NEXT: AtomicCompareExchange [[#int]] [[#Result2:]] [[#Pointer]] [[#DeviceScope]] [[#SequentiallyConsistent_MS]] [[#SequentiallyConsistent_MS]] [[#]] [[#Comparator]]
; CHECK-NEXT: Store [[#]] [[#Result2]]
; CHECK-NEXT: IEqual [[#bool]] [[#]] [[#Result2]] [[#Comparator]]
  %call11 = call spir_func i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS3VU7_AtomiciPii12memory_orderS4_12memory_scope(ptr addrspace(3) %arrayidx7, ptr null, i32 14, i32 0, i32 0, i32 0)
  ret void
}

declare spir_func i1 @_Z37atomic_compare_exchange_weak_explicitPU3AS3VU7_AtomiciPii12memory_orderS4_12memory_scope(ptr addrspace(3), ptr, i32, i32, i32, i32)
declare spir_func i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS3VU7_AtomiciPii12memory_orderS4_12memory_scope(ptr addrspace(3), ptr, i32, i32, i32, i32)

!opencl.ocl.version = !{!0}

!0 = !{i32 3, i32 0}

