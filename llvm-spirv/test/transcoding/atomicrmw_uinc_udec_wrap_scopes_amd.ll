; Check that scope value survives the round-trip and aligns with the triple.

; RUN: llvm-spirv %s -o %t.spv
; RUN: spirv-val %t.spv

; RUN: llvm-spirv -r --spirv-target-triple=amdgcn-amd-amdhsa %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-AMDGCN

; SPV-IR mode: representation independent.
; RUN: llvm-spirv -r --spirv-target-env=SPV-IR --spirv-target-triple=amdgcn-amd-amdhsa %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-AMDGCN

; AMD vendor but not AMDGCN arch: lowered, generic names.
; RUN: llvm-spirv -r --spirv-target-triple=spirv64-amd-amdhsa %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-DEFAULT

; Non-AMD target: the helper calls are left alone.
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-NOLOWER --implicit-check-not=atomicrmw
; RUN: llvm-spirv -r --spirv-target-triple=spirv64-unknown-unknown %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-NOLOWER --implicit-check-not=atomicrmw

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv64-amd-amdhsa"

@ui = common dso_local addrspace(1) global i32 0, align 4

; CHECK-AMDGCN: %[[#R:]] = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 syncscope("agent") seq_cst
; CHECK-AMDGCN-NEXT: ret i32 %[[#R]]
; CHECK-DEFAULT: %[[#R:]] = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 syncscope("device") seq_cst
; CHECK-DEFAULT-NEXT: ret i32 %[[#R]]
; CHECK-NOLOWER: %r = call {{.*}}@__translate_spirv_atomic_uinc_wrap_p1_i32
; CHECK-NOLOWER-NEXT: ret i32 %r
define spir_func i32 @test_uinc_wrap_device_scope() {
entry:
  %r = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 syncscope("device") seq_cst
  ret i32 %r
}

; CHECK-AMDGCN: %[[#R:]] = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 syncscope("wavefront") seq_cst
; CHECK-AMDGCN-NEXT: ret i32 %[[#R]]
; CHECK-DEFAULT: %[[#R:]] = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 syncscope("subgroup") seq_cst
; CHECK-DEFAULT-NEXT: ret i32 %[[#R]]
; CHECK-NOLOWER: %r = call {{.*}}@__translate_spirv_atomic_udec_wrap_p1_i32
; CHECK-NOLOWER-NEXT: ret i32 %r
define spir_func i32 @test_udec_wrap_subgroup_scope() {
entry:
  %r = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 syncscope("subgroup") seq_cst
  ret i32 %r
}
