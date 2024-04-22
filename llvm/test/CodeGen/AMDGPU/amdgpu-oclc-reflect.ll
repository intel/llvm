; RUN: opt -S -p amdgpu-oclc-reflect %s | FileCheck %s -check-prefixes=CHECK,CHECK-SAFE-ATOMICS
; RUN: opt -S -p amdgpu-oclc-reflect -amdgpu-oclc-unsafe-int-atomics=true %s | FileCheck %s -check-prefixes=CHECK,CHECK-UNSAFE-ATOMICS

target triple = "amdgcn-amd-amdhsa"

@.str = private unnamed_addr addrspace(4) constant [31 x i8] c"AMDGPU_OCLC_UNSAFE_INT_ATOMICS\00", align 1

declare hidden i32 @__oclc_amdgpu_reflect(ptr addrspace(4) noundef) local_unnamed_addr

define i32 @foo() {
; CHECK-NOT: call i32 @__oclc_amdgpu_reflect(ptr addrspace(4) noundef @.str)
; CHECK-SAFE-ATOMICS: ret i32 0
; CHECK-UNSAFE-ATOMICS: ret i32 1
  %call = tail call i32 @__oclc_amdgpu_reflect(ptr addrspace(4) noundef @.str)
  ret i32 %call
}
