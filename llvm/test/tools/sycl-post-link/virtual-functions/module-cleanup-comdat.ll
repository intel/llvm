; RUN: sycl-post-link -split=auto -S < %s -o %t.table
;
; Virtual functions cleanup drops their bodies from some of device images
; turning them into declarations, but declarations can't have "comdat"
; attached to them, so this test ensures that we can handle "comdat" without
; crashes.

; RUN: FileCheck %s --input-file=%t_0.ll --check-prefix=CHECK-IR0
; RUN: FileCheck %s --input-file=%t_1.ll --check-prefix=CHECK-IR1

; CHECK-IR0: define spir_func void @foo
; CHECK-IR1-DAG: declare spir_func void @foo
; CHECK-IR1-DAG: define weak_odr dso_local spir_kernel void @kernel

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

$foo = comdat any

@vtable = linkonce_odr dso_local unnamed_addr addrspace(1) constant { [3 x ptr addrspace(4)] } { [3 x ptr addrspace(4)] [ptr addrspace(4) null, ptr addrspace(4) null, ptr addrspace(4) addrspacecast (ptr @foo to ptr addrspace(4))] }, align 8

define linkonce_odr spir_func void @foo() #0 comdat {
entry:
  ret void
}

define weak_odr dso_local spir_kernel void @kernel(ptr addrspace(1) noundef align 8 %_arg_StorageAcc) #1 {
entry:
  store ptr addrspace(1) getelementptr inbounds inrange(-16, 8) (i8, ptr addrspace(1) @vtable, i64 16), ptr addrspace(1) %_arg_StorageAcc, align 8
  ret void
}

attributes #0 = { "indirectly-callable"="set-foo" "sycl-module-id"="v.cpp" }
attributes #1 = { "sycl-module-id"="v.cpp" }

