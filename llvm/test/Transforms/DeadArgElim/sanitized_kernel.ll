; RUN: opt < %s -passes=deadargelim -S | FileCheck %s

; CHECK: define weak_odr dso_local spir_kernel void @kernel(ptr addrspace(1) %__asan_launch)
define weak_odr dso_local spir_kernel void @kernel(ptr addrspace(1) %__asan_launch) #1 {
  ret void
}

attributes #1 = { sanitize_address }
