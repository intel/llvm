; Test --spirv-addrspace-map address space remapping on SPIR-V -> LLVM translation.

; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -r %t.spv --spirv-addrspace-map=0:4,1:1,2:2,3:3,4:0 \
; RUN:   -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-MAPPED
; RUN: llvm-spirv -r %t.spv \
; RUN:   -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-IDENTITY
; RUN: llvm-spirv -r %t.spv --spirv-addrspace-map=0:4 \
; RUN:   -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-PARTIAL
; RUN: llvm-spirv -r %t.spv --spirv-addrspace-map=1:5 \
; RUN:   -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-GLOBAL
; RUN: not llvm-spirv -r %t.spv --spirv-addrspace-map=99:0 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-ERR-IDX
; RUN: not llvm-spirv -r %t.spv --spirv-addrspace-map=bad 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-ERR-FMT

; CHECK-ERR-IDX: not a valid SPIRAddressSpace index
; CHECK-ERR-FMT: Invalid format for --spirv-addrspace-map

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir64-unknown-unknown"

; Global variable: lives in SPIRAS_Global (1) by default.
; With 1->5 mapping it must move to addrspace(5); all other cases keep AS1.
; CHECK-MAPPED: @gv = {{.*}}addrspace(1){{.*}}global i32
; CHECK-IDENTITY: @gv = {{.*}}addrspace(1){{.*}}global i32
; CHECK-PARTIAL: @gv = {{.*}}addrspace(1){{.*}}global i32
; CHECK-GLOBAL: @gv = {{.*}}addrspace(5){{.*}}global i32
@gv = addrspace(1) global i32 0, align 4

; CHECK-MAPPED: define{{.*}} @test_stable_and_generic({{.*}}ptr addrspace(1){{.*}}ptr addrspace(3){{.*}}ptr{{( addrspace\(0\))?}}
; CHECK-IDENTITY: define{{.*}} @test_stable_and_generic({{.*}}ptr addrspace(1){{.*}}ptr addrspace(3){{.*}}ptr addrspace(4)
; CHECK-PARTIAL: define{{.*}} @test_stable_and_generic({{.*}}ptr addrspace(1){{.*}}ptr addrspace(3){{.*}}ptr addrspace(4)
; CHECK-GLOBAL: define{{.*}} @test_stable_and_generic({{.*}}ptr addrspace(5){{.*}}ptr addrspace(3){{.*}}ptr addrspace(4)
define spir_kernel void @test_stable_and_generic(ptr addrspace(1) %global_p,
                                                 ptr addrspace(3) %local_p,
                                                 ptr addrspace(4) %generic_p) {
  ret void
}

; CHECK-MAPPED: define{{.*}} @test_constant({{.*}}ptr addrspace(2)
; CHECK-IDENTITY: define{{.*}} @test_constant({{.*}}ptr addrspace(2)
; CHECK-PARTIAL: define{{.*}} @test_constant({{.*}}ptr addrspace(2)
define spir_kernel void @test_constant(ptr addrspace(2) %const_p) {
  ret void
}

; CHECK-MAPPED: define{{.*}} @test_private(
; CHECK-MAPPED: alloca i32,{{.*}} addrspace(4)
; CHECK-IDENTITY: define{{.*}} @test_private(
; CHECK-IDENTITY: alloca i32, align
; CHECK-PARTIAL: define{{.*}} @test_private(
; CHECK-PARTIAL: alloca i32,{{.*}} addrspace(4)
define spir_func i32 @test_private() {
  %x = alloca i32
  %v = load i32, ptr %x
  ret i32 %v
}

attributes #0 = { nounwind }
