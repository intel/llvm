; RUN: opt -S -passes=sycl-virtual-functions-analysis %s | FileCheck %s
;
; Virtual functions may be defined in a different translation unit (or even in a
; different device image) than the one which constructs objects of the
; corresponding polymorphic class. In that case a vtable references only a
; declaration of a virtual function, but the "indirectly-callable" attribute is
; still attached to it and therefore "calls-indirectly" should still be
; propagated onto kernels using that vtable.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

@vtable = linkonce_odr dso_local unnamed_addr addrspace(1) constant { [3 x ptr addrspace(4)] } { [3 x ptr addrspace(4)] [ptr addrspace(4) null, ptr addrspace(4) null, ptr addrspace(4) addrspacecast (ptr @foo to ptr addrspace(4))] }, align 8

declare spir_func void @foo() #0

define weak_odr dso_local spir_kernel void @kernel(ptr addrspace(1) noundef align 8 %_arg_StorageAcc) #1 {
entry:
  store ptr addrspace(1) getelementptr inbounds inrange(-16, 8) (i8, ptr addrspace(1) @vtable, i64 16), ptr addrspace(1) %_arg_StorageAcc, align 8
  ret void
}

; CHECK: @kernel({{.*}} #[[#KERNEL_ATTRS:]]
; CHECK: attributes #[[#KERNEL_ATTRS]] = {{.*}}"calls-indirectly"="set-foo"

attributes #0 = { "indirectly-callable"="set-foo" "sycl-module-id"="v.cpp" }
attributes #1 = { "sycl-module-id"="v.cpp" }
