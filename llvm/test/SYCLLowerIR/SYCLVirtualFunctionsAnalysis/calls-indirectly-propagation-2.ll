; RUN: opt -S -passes=sycl-virtual-functions-analysis %s | FileCheck %s
;
; This is a more complicated version of a test intended to check that if a
; kernel uses a vtable, then it should be annotated with an attribute
; "calls-indirectly" that has the same value as a function referenced by that
; vtable.
; Main thing which is tested here is case when a kernel ends up using more than
; one set of virtual functions.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

@vtable_foo = linkonce_odr dso_local unnamed_addr addrspace(1) constant { [3 x ptr addrspace(4)] } { [3 x ptr addrspace(4)] [ptr addrspace(4) null, ptr addrspace(4) null, ptr addrspace(4) addrspacecast (ptr @foo to ptr addrspace(4))] }, align 8
@vtable_bar = linkonce_odr dso_local unnamed_addr addrspace(1) constant { [3 x ptr addrspace(4)] } { [3 x ptr addrspace(4)] [ptr addrspace(4) null, ptr addrspace(4) null, ptr addrspace(4) addrspacecast (ptr @bar to ptr addrspace(4))] }, align 8

define linkonce_odr spir_func void @foo() #0 {
entry:
  ret void
}

define linkonce_odr spir_func void @bar() #1 {
entry:
  ret void
}

define weak_odr dso_local spir_kernel void @kernel(ptr addrspace(1) noundef align 8 %_arg_StorageAcc) #2 {
entry:
  store ptr addrspace(1) getelementptr inbounds inrange(-16, 8) (i8, ptr addrspace(1) @vtable_foo, i64 16), ptr addrspace(1) %_arg_StorageAcc, align 8
  store ptr addrspace(1) getelementptr inbounds inrange(-16, 8) (i8, ptr addrspace(1) @vtable_bar, i64 16), ptr addrspace(1) %_arg_StorageAcc, align 8
  ret void
}

define weak_odr dso_local spir_kernel void @kernel_already_uses(ptr addrspace(1) noundef align 8 %_arg_StorageAcc) #3 {
entry:
  store ptr addrspace(1) getelementptr inbounds inrange(-16, 8) (i8, ptr addrspace(1) @vtable_bar, i64 16), ptr addrspace(1) %_arg_StorageAcc, align 8
  ret void
}

; CHECK: @kernel({{.*}} #[[#KERNEL_ATTRS:]]
; CHECK: @kernel_already_uses({{.*}} #[[#KERNEL_ATTRS]]
;
; CHECK: attributes #[[#KERNEL_ATTRS]] = {{.*}}"calls-indirectly"="set-foo,set-bar"

attributes #0 = { "indirectly-callable"="set-foo" "sycl-module-id"="v.cpp" }
attributes #1 = { "indirectly-callable"="set-bar" "sycl-module-id"="v.cpp" }
attributes #2 = { "sycl-module-id"="v.cpp" }
attributes #3 = { "calls-indirectly"="set-foo" "sycl-module-id"="v.cpp" }
