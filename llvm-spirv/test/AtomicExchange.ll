; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r --spirv-target-env=SPV-IR %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-SPV-IR
; RUN: llvm-spirv -r  %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-OCL-IR

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

; CHECK-DAG: TypeInt [[#Int64:]] 64 0
; CHECK-COUNT-9: AtomicExchange [[#Int64]] [[#]] [[#]] [[#]] [[#]] [[#]]

; CHECK-SPV-IR-COUNT-9: __spirv_AtomicExchange
; CHECK-OCL-IR-COUNT-9: call spir_func i64 @_Z9atom_xchg
; TODO: The call to @_Z9atom_xchg is not yet lowered to the corresponding SPIR-V instruction.
;       It currently remains as a plain function call during SPIR-V generation.

%Type1 = type { i64 }
%Type2 = type { ptr addrspace(4) }

define linkonce_odr dso_local spir_func void @f1() {
entry:
  %a = alloca %Type1, align 8
  %a.ascast = addrspacecast ptr %a to ptr addrspace(4)
  %p = ptrtoint ptr addrspace(4) %a.ascast to i64
  %res = call spir_func i64 @_Z22__spirv_AtomicExchangePyN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEy(ptr addrspace(4) %a.ascast, i32 0, i32 912, i64 3)
  ret void
}

define linkonce_odr dso_local spir_func void @f2() {
entry:
  %a = alloca %Type1, align 8
  %a.ascast = addrspacecast ptr %a to ptr addrspace(4)
  %p = ptrtoint ptr addrspace(4) %a.ascast to i64
  %res = call spir_func i64 @_Z22__spirv_AtomicExchangePxN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEx(ptr addrspace(4) %a.ascast, i32 0, i32 912, i64 3)
  ret void
}

define linkonce_odr dso_local spir_func void @f3() {
entry:
  %a = alloca %Type1, align 8
  %a.ascast = addrspacecast ptr %a to ptr addrspace(4)
  %p = ptrtoint ptr addrspace(4) %a.ascast to i64
  %res = call spir_func i64 @_Z22__spirv_AtomicExchangePmN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEm(ptr addrspace(4) %a.ascast, i32 0, i32 912, i64 3)
  ret void
}

define linkonce_odr dso_local spir_func void @f4() {
entry:
  %a = alloca %Type1, align 8
  %a.ascast = addrspacecast ptr %a to ptr addrspace(4)
  %p = ptrtoint ptr addrspace(4) %a.ascast to i64
  %res = call spir_func i64 @_Z22__spirv_AtomicExchangePlN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEl(ptr addrspace(4) %a.ascast, i32 0, i32 912, i64 3)
  ret void
}


define linkonce_odr dso_local spir_func void @f5() {
entry:
  %a = alloca %Type2, align 8
  %a.ascast = addrspacecast ptr %a to ptr addrspace(4)
  %p = ptrtoint ptr addrspace(4) %a.ascast to i64
  %res = call spir_func i64 @_Z22__spirv_AtomicExchangePmN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEm(ptr addrspace(4) %a.ascast, i32 0, i32 912, i64 %p)
  ret void
}

define linkonce_odr dso_local spir_func void @f6() {
entry:
  %a = alloca %Type2, align 8
  %a.ascast = addrspacecast ptr %a to ptr addrspace(4)
  %p = ptrtoint ptr addrspace(4) %a.ascast to i64
  %res = call spir_func i64 @_Z22__spirv_AtomicExchangePmN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEm(ptr addrspace(4) %a.ascast, i32 0, i32 912, i64 %p)
  ret void
}

define linkonce_odr dso_local spir_func void @f7() {
entry:
  %a = alloca %Type2, align 8
  %a.ascast = addrspacecast ptr %a to ptr addrspace(4)
  %p = ptrtoint ptr addrspace(4) %a.ascast to i64
  %res = call spir_func i64 @_Z22__spirv_AtomicExchangePmN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEm(ptr addrspace(4) %a.ascast, i32 0, i32 912, i64 %p)
  ret void
}

define linkonce_odr dso_local spir_func void @f8() {
entry:
  %a = alloca %Type2, align 8
  %a.ascast = addrspacecast ptr %a to ptr addrspace(4)
  %p = ptrtoint ptr addrspace(4) %a.ascast to i64
  %res = call spir_func i64 @_Z22__spirv_AtomicExchangePmN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEm(ptr addrspace(4) %a.ascast, i32 0, i32 912, i64 %p)
  ret void
}

; Function Attrs: convergent mustprogress norecurse nounwind
define linkonce_odr dso_local spir_func void @f9() {
entry:
  %a = alloca %Type2, align 8
  %a.ascast = addrspacecast ptr %a to ptr addrspace(4)
  %p = ptrtoint ptr addrspace(4) %a.ascast to i64
  %res = call spir_func i64 @_Z22__spirv_AtomicExchangePmN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEm(ptr addrspace(4) %a.ascast, i32 0, i32 912, i64 %p)
  ret void
}

declare dso_local spir_func i64 @_Z22__spirv_AtomicExchangePyN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEy(ptr addrspace(4), i32, i32, i64)
declare dso_local spir_func i64 @_Z22__spirv_AtomicExchangePxN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEx(ptr addrspace(4), i32, i32, i64)
declare dso_local spir_func i64 @_Z22__spirv_AtomicExchangePmN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEm(ptr addrspace(4), i32, i32, i64)
declare dso_local spir_func i64 @_Z22__spirv_AtomicExchangePlN5__spv5Scope4FlagENS0_19MemorySemanticsMask4FlagEl(ptr addrspace(4), i32, i32, i64)
