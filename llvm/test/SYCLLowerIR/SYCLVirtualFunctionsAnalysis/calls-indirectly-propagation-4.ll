; RUN: opt -S -passes=sycl-virtual-functions-analysis %s | FileCheck %s
;
; This is a more complicated version of a test intended to check that if a
; kernel uses a vtable, then it should be annotated with an attribute
; "calls-indirectly" that has the same value as a function referenced by that
; vtable.
; This test case is focused on more complex call graph of a kernel, where vtable
; operation is not performed directly by a kernel, but instead is done by some
; other helper function.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

@vtable = linkonce_odr dso_local unnamed_addr addrspace(1) constant { [4 x ptr addrspace(4)] } { [4 x ptr addrspace(4)] [ptr addrspace(4) null, ptr addrspace(4) null, ptr addrspace(4) addrspacecast (ptr @foo to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr @bar to ptr addrspace(4))] }, align 8

define linkonce_odr spir_func void @foo() #0 {
entry:
  ret void
}

define linkonce_odr spir_func void @bar() #1 {
entry:
  ret void
}

define internal spir_func void @helper(ptr addrspace(1) noundef align 8 %arg) {
entry:
  store ptr addrspace(1) getelementptr inbounds inrange(-16, 8) (i8, ptr addrspace(1) @vtable, i64 16), ptr addrspace(1) %arg, align 8
  ret void
}

define weak_odr dso_local spir_kernel void @kernel(ptr addrspace(1) noundef align 8 %_arg_StorageAcc) #2 {
entry:
  call void @helper(ptr addrspace(1) %_arg_StorageAcc)
  ret void
}

define internal spir_func void @helper2(ptr addrspace(1) noundef align 8 %arg) {
entry:
  call void @helper(ptr addrspace(1) %arg)
  ret void
}

define weak_odr dso_local spir_kernel void @kernel2(ptr addrspace(1) noundef align 8 %_arg_StorageAcc) #2 {
entry:
  call void @helper2(ptr addrspace(1) %_arg_StorageAcc)
  ret void
}

; CHECK: @kernel{{.*}} #[[#KERNEL_ATTRS:]]
; CHECK: @kernel2{{.*}} #[[#KERNEL_ATTRS]]
;
; CHECK: attributes #[[#KERNEL_ATTRS]] = {{.*}}"calls-indirectly"="set-foo,set-bar"

attributes #0 = { "indirectly-callable"="set-foo" "sycl-module-id"="v.cpp" }
attributes #1 = { "indirectly-callable"="set-bar" "sycl-module-id"="v.cpp" }
attributes #2 = { "sycl-module-id"="v.cpp" }

