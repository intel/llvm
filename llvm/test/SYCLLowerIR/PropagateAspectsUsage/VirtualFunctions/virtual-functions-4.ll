; RUN: opt -passes=sycl-propagate-aspects-usage < %s -S | FileCheck %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

@vtable = linkonce_odr dso_local unnamed_addr addrspace(1) constant { [3 x ptr addrspace(4)] } { [3 x ptr addrspace(4)] [ptr addrspace(4) null, ptr addrspace(4) null, ptr addrspace(4) addrspacecast (ptr @foo to ptr addrspace(4))] }, align 8

; CHECK: @foo() #0 !sycl_used_aspects ![[#aspects:]]
define linkonce_odr spir_func void @foo() #0 {
entry:
  %tmp = alloca double
  ret void
}

; "Construction" kernels which reference vtables but do not actually
; perform any virtual calls should not have aspects propagated to them.
; CHECK-NOT: @construct({{.*}}){{.*}}!sycl_used_aspects
define weak_odr dso_local spir_kernel void @construct(ptr addrspace(1) noundef align 8 %_arg_StorageAcc) {
entry:
  store ptr addrspace(1) getelementptr inbounds inrange(-16, 8) (i8, ptr addrspace(1) @vtable, i64 16), ptr addrspace(1) %_arg_StorageAcc, align 8
  ret void
}

; Note: after SYCLVirtualFunctionAnalysis pass, the construction kernels will have 
; "calls-indirectly" attribute, but even so they should not have aspects propagated 
; to them (as the construction kernels have no virtual calls).
; CHECK-NOT: @construct2({{.*}}){{.*}}!sycl_used_aspects
define weak_odr dso_local spir_kernel void @construct2(ptr addrspace(1) noundef align 8 %_arg_StorageAcc) #1 {
entry:
  store ptr addrspace(1) getelementptr inbounds inrange(-16, 8) (i8, ptr addrspace(1) @vtable, i64 16), ptr addrspace(1) %_arg_StorageAcc, align 8
  ret void
}

; CHECK: ![[#aspects]] = !{i32 6}

attributes #0 = { "indirectly-callable"="set-foo" }
attributes #1 = { "calls-indirectly"="set-foo" }

!sycl_aspects = !{!0}
!0 = !{!"fp64", i32 6}