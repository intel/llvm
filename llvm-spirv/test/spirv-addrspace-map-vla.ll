; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_variable_length_array -o %t.spv
; RUN: llvm-spirv -r %t.spv --spirv-addrspace-map=0:4,1:1,2:2,3:3,4:0 \
; RUN:   -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-MAPPED
; RUN: llvm-spirv -r %t.spv \
; RUN:   -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-IDENTITY
; RUN: llvm-spirv -r %t.spv --spirv-addrspace-map=0:4 \
; RUN:   -o - | llvm-dis | FileCheck %s --check-prefix=CHECK-PARTIAL

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir64-unknown-unknown"

; Test --spirv-addrspace-map address space remapping for functions using
; variable-length arrays (requires SPV_INTEL_variable_length_array), this
; triggers generation of alloca and @llvm.stacksave/restore which need to be
; emitted with correct addess space.
; CHECK-MAPPED: define{{.*}} @test_stack_save_restore(
; CHECK-MAPPED: call addrspace(4) ptr addrspace(4) @llvm.stacksave.p4()
; CHECK-MAPPED: alloca i32, i64 %n, align 4, addrspace(4)
; CHECK-MAPPED: call addrspace(4) void @llvm.stackrestore.p4(ptr addrspace(4)
; CHECK-IDENTITY: define{{.*}} @test_stack_save_restore(
; CHECK-IDENTITY: call ptr @llvm.stacksave.p0()
; CHECK-IDENTITY: alloca i32, i64 %n, align 4{{$}}
; CHECK-IDENTITY: call void @llvm.stackrestore.p0(ptr
; CHECK-PARTIAL: define{{.*}} @test_stack_save_restore(
; CHECK-PARTIAL: call addrspace(4) ptr addrspace(4) @llvm.stacksave.p4()
; CHECK-PARTIAL: alloca i32, i64 %n, align 4, addrspace(4)
; CHECK-PARTIAL: call addrspace(4) void @llvm.stackrestore.p4(ptr addrspace(4)
define spir_func void @test_stack_save_restore(i64 %n) {
  %saved = call ptr @llvm.stacksave.p0()
  %vla = alloca i32, i64 %n, align 4
  store i32 0, ptr %vla, align 4
  call void @llvm.stackrestore.p0(ptr %saved)
  ret void
}

declare ptr @llvm.stacksave.p0() #0
declare void @llvm.stackrestore.p0(ptr) #0

attributes #0 = { nounwind }
