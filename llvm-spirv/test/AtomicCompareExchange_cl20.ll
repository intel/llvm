; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; CHECK: 3 Source 3 200000

; Int64Atomics capability must be declared only if atomic builtins have 64-bit integers arguments.
; CHECK-NOT: Capability Int64Atomics

; CHECK: 4 TypeInt [[int:[0-9]+]] 32 0
; CHECK: Constant [[int]] [[DeviceScope:[0-9]+]] 1
; CHECK: Constant [[int]] [[SequentiallyConsistent_MS:[0-9]+]] 16
; CHECK: 4 TypePointer [[int_ptr:[0-9]+]] 8 [[int]]
; CHECK: 2 TypeBool [[bool:[0-9]+]]

; Function Attrs: nounwind
define spir_func void @test(ptr addrspace(4) %object, ptr addrspace(4) %expected, i32 %desired) #0 {
; CHECK: FunctionParameter [[int_ptr]] [[object:[0-9]+]]
; CHECK: FunctionParameter [[int_ptr]] [[expected:[0-9]+]]
; CHECK: FunctionParameter [[int]] [[desired:[0-9]+]]

entry:
  %object.addr = alloca ptr addrspace(4), align 4
  %expected.addr = alloca ptr addrspace(4), align 4
  %desired.addr = alloca i32, align 4
  %strong_res = alloca i8, align 1
  %res = alloca i8, align 1
  %weak_res = alloca i8, align 1
  store ptr addrspace(4) %object, ptr %object.addr, align 4
  store ptr addrspace(4) %expected, ptr %expected.addr, align 4
  store i32 %desired, ptr %desired.addr, align 4
  %0 = load ptr addrspace(4), ptr %object.addr, align 4
  %1 = load ptr addrspace(4), ptr %expected.addr, align 4
  %2 = load i32, ptr %desired.addr, align 4
; CHECK: Store [[object_addr:[0-9]+]] [[object]]
; CHECK: Store [[expected_addr:[0-9]+]] [[expected]]
; CHECK: Store [[desired_addr:[0-9]+]] [[desired]]
; CHECK: Load [[int_ptr]] [[Pointer:[0-9]+]] [[object_addr]]
; CHECK: Load [[int_ptr]] [[exp:[0-9]+]] [[expected_addr]]
; CHECK: Load [[int]] [[Value:[0-9]+]] [[desired_addr]]
; CHECK: Load [[int]] [[Comparator:[0-9]+]] [[exp]]
; CHECK-NEXT: 9 AtomicCompareExchange [[int]] [[Result:[0-9]+]] [[Pointer]] [[DeviceScope]] [[SequentiallyConsistent_MS]] [[SequentiallyConsistent_MS]] [[Value]] [[Comparator]]
  %call = call spir_func zeroext i1 @_Z30atomic_compare_exchange_strongPVU3AS4U7_AtomiciPU3AS4ii(ptr addrspace(4) %0, ptr addrspace(4) %1, i32 %2)
; CHECK-NEXT: Store [[exp]] [[Result]]
; CHECK-NEXT: IEqual [[bool]] [[CallRes:[0-9]+]] [[Result]] [[Comparator]]
; CHECK-NOT: [[Result]]
  %frombool = zext i1 %call to i8
  store i8 %frombool, ptr %strong_res, align 1
  %3 = load i8, ptr %strong_res, align 1
  %tobool = trunc i8 %3 to i1
  %lnot = xor i1 %tobool, true
  %frombool1 = zext i1 %lnot to i8
  store i8 %frombool1, ptr %res, align 1
  %4 = load ptr addrspace(4), ptr %object.addr, align 4
  %5 = load ptr addrspace(4), ptr %expected.addr, align 4
  %6 = load i32, ptr %desired.addr, align 4

; CHECK: Load [[int_ptr]] [[Pointer:[0-9]+]] [[object_addr]]
; CHECK: Load [[int_ptr]] [[exp:[0-9]+]] [[expected_addr]]
; CHECK: Load [[int]] [[Value:[0-9]+]] [[desired_addr]]
; CHECK: Load [[int]] [[ComparatorWeak:[0-9]+]] [[exp]]
  %call2 = call spir_func zeroext i1 @_Z28atomic_compare_exchange_weakPVU3AS4U7_AtomiciPU3AS4ii(ptr addrspace(4) %4, ptr addrspace(4) %5, i32 %6)
; CHECK-NEXT: 9 AtomicCompareExchange [[int]] [[Result:[0-9]+]] [[Pointer]] [[DeviceScope]] [[SequentiallyConsistent_MS]] [[SequentiallyConsistent_MS]] [[Value]] [[ComparatorWeak]]
; CHECK-NEXT: Store [[exp]] [[Result]]
; CHECK-NEXT: IEqual [[bool]] [[CallRes:[0-9]+]] [[Result]] [[ComparatorWeak]]
; CHECK-NOT: [[Result]]

  %frombool3 = zext i1 %call2 to i8
  store i8 %frombool3, ptr %weak_res, align 1
  %7 = load i8, ptr %weak_res, align 1
  %tobool4 = trunc i8 %7 to i1
  %lnot5 = xor i1 %tobool4, true
  %frombool6 = zext i1 %lnot5 to i8
  store i8 %frombool6, ptr %res, align 1
  ret void
}

declare spir_func zeroext i1 @_Z30atomic_compare_exchange_strongPVU3AS4U7_AtomiciPU3AS4ii(ptr addrspace(4), ptr addrspace(4), i32) #1

declare spir_func zeroext i1 @_Z28atomic_compare_exchange_weakPVU3AS4U7_AtomiciPU3AS4ii(ptr addrspace(4), ptr addrspace(4), i32) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!opencl.compiler.options = !{!2}


!0 = !{i32 1, i32 2}
!1 = !{i32 2, i32 0}
!2 = !{}
