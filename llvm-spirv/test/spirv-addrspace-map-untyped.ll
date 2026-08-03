; Test that --spirv-addrspace-map applies correctly when untyped pointers
; (SPV_KHR_untyped_pointers) are used.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_untyped_pointers -o %t.untyped.spv
; RUN: llvm-spirv -r %t.untyped.spv --spirv-addrspace-map=0:4,1:1,2:2,3:3,4:0 \
; RUN:   -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-MAPPED
; RUN: llvm-spirv -r %t.untyped.spv \
; RUN:   -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-IDENTITY

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir64-unknown-unknown"

; With mapping 4->0, the generic AS4 pointer remaps to AS0 (printed as bare ptr).
; CHECK-MAPPED: define{{.*}} @test_generic_atomic(
; CHECK-MAPPED: call spir_func addrspace(4) i32 @_Z10atomic_addPVii(ptr

; Without mapping, generic stays at AS4.
; CHECK-IDENTITY: define{{.*}} @test_generic_atomic(
; CHECK-IDENTITY: call spir_func i32 @_Z10atomic_addPU3AS4Vii(ptr addrspace(4)

define spir_func i32 @test_generic_atomic(ptr addrspace(4) %p) {
  %v = load atomic i32, ptr addrspace(4) %p seq_cst, align 4
  ret i32 %v
}

attributes #0 = { nounwind }
