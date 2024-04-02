; ModuleID = ''
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r --spirv-target-env=CL2.0 %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s

; Check 'LLVM ==> SPIR-V ==> LLVM' conversion of atomic_compare_exchange_strong and atomic_compare_exchange_weak.

; SPIR-V does not include an equivalent of atomic_compare_exchange_weak
; (OpAtomicCompareExchangeWeak is identical to OpAtomicCompareExchange and
; is deprecated, and removed in SPIR-V 1.4.)
; This breaks the round trip for atomic_compare_exchange_weak, which must be
; translated back to LLVM IR as atomic_compare_exchange_strong, regardless
; of whether OpAtomicCompareExchange or OpAtomicCompareExchangeWeak is used.

; Function Attrs: nounwind

; CHECK-LABEL:   define spir_func void @test_strong
; CHECK-NEXT:    entry:
; CHECK:         [[PTR_STRONG:%expected[0-9]*]] = alloca i32, align 4
; CHECK:         store i32 {{.*}}, ptr [[PTR_STRONG]]
; CHECK:         [[PTR_STRONG]].as = addrspacecast ptr [[PTR_STRONG]] to ptr addrspace(4)
; CHECK:         call spir_func i1 @_Z39atomic_compare_exchange_strong_explicit{{.*}}(ptr {{.*}} %object, ptr {{.*}} [[PTR_STRONG]].as, i32 %desired, i32 5, i32 5, i32 2)
; CHECK:         load i32, ptr addrspace(4) [[PTR_STRONG]].as

; CHECK-LABEL:   define spir_func void @test_weak
; CHECK-NEXT:    entry:
; CHECK:         [[PTR_WEAK:%expected[0-9]*]] = alloca i32, align 4
; CHECK:         store i32 {{.*}}, ptr [[PTR_WEAK]]
; CHECK:         [[PTR_WEAK]].as = addrspacecast ptr [[PTR_WEAK]] to ptr addrspace(4)
; CHECK:         call spir_func i1 @_Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope{{.*}}(ptr {{.*}} %object, ptr {{.*}} [[PTR_WEAK]].as, i32 %desired, i32 5, i32 5, i32 2)
; CHECK:         load i32, ptr addrspace(4) [[PTR_WEAK]].as

; Check that alloca for atomic_compare_exchange is being created in the entry block.

; CHECK-LABEL:   @atomic_in_loop
; CHECK-LABEL:   entry:
; CHECK:         %expected{{[0-9]*}} = alloca i32
; CHECK-LABEL:   for.body:
; CHECK-NOT:     %expected{{[0-9]*}} = alloca i32
; CHECK:         call spir_func i1 @_Z39atomic_compare_exchange_strong_explicit{{.*}}(ptr {{.*}} {{.*}}, ptr addrspace(4) {{.*}}, i32 {{.*}}, i32 5, i32 5, i32 2)

; Function Attrs: nounwind
define spir_func void @test_strong(ptr addrspace(4) %object, ptr addrspace(4) %expected, i32 %desired) #0 {
entry:
  %call = tail call spir_func zeroext i1 @_Z30atomic_compare_exchange_strongPVU3AS4U7_AtomiciPU3AS4ii(ptr addrspace(4) %object, ptr addrspace(4) %expected, i32 %desired) #2
  ret void
}

declare spir_func zeroext i1 @_Z30atomic_compare_exchange_strongPVU3AS4U7_AtomiciPU3AS4ii(ptr addrspace(4), ptr addrspace(4), i32) #1

; Function Attrs: nounwind
define spir_func void @test_weak(ptr addrspace(4) %object, ptr addrspace(4) %expected, i32 %desired) #0 {
entry:
  %call2 = tail call spir_func zeroext i1 @_Z28atomic_compare_exchange_weakPVU3AS4U7_AtomiciPU3AS4ii(ptr addrspace(4) %object, ptr addrspace(4) %expected, i32 %desired) #2
  ret void
}

declare spir_func zeroext i1 @_Z28atomic_compare_exchange_weakPVU3AS4U7_AtomiciPU3AS4ii(ptr addrspace(4), ptr addrspace(4), i32) #1

; Function Attrs: nounwind
define spir_kernel void @atomic_in_loop(ptr addrspace(1) %destMemory, ptr addrspace(1) %oldValues) #0 {
entry:
  %destMemory.addr = alloca ptr addrspace(1), align 8
  %oldValues.addr = alloca ptr addrspace(1), align 8
  %expected = alloca i32, align 4
  %previous = alloca i32, align 4
  %i = alloca i32, align 4
  store ptr addrspace(1) %destMemory, ptr %destMemory.addr, align 8
  store ptr addrspace(1) %oldValues, ptr %oldValues.addr, align 8
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 100000
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load ptr addrspace(1), ptr %destMemory.addr, align 8
  %2 = addrspacecast ptr addrspace(1) %1 to ptr addrspace(4)
  %3 = addrspacecast ptr %expected to ptr addrspace(4)
  %4 = load ptr addrspace(1), ptr %oldValues.addr, align 8
  %5 = load i32, ptr addrspace(1) %4, align 4
  %call = call spir_func zeroext i1 @_Z30atomic_compare_exchange_strongPVU3AS4U7_AtomiciPU3AS4ii(ptr addrspace(4) %2, ptr addrspace(4) %3, i32 %5)
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %6 = load i32, ptr %i, align 4
  %inc = add nsw i32 %6, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}

!0 = !{i32 1, i32 2}
!1 = !{i32 2, i32 0}
!2 = !{}
