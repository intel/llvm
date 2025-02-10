; RUN: opt < %s -passes=msan -msan-instrumentation-with-call-threshold=0 -msan-eager-checks=1 -S | FileCheck %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

@.str = external addrspace(1) constant [59 x i8]
@__spirv_BuiltInGlobalInvocationId = external addrspace(1) constant <3 x i64>
@dev_global_no_users = dso_local addrspace(1) global { [4 x i32] } zeroinitializer

define spir_func void @_ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlvE_clEv() {
entry:
  %call = call spir_func ptr addrspace(4) null(ptr addrspace(4) addrspacecast (ptr addrspace(1) @.str to ptr addrspace(4)), i64 0)
  ret void
}

; CHECK: @__MsanDeviceGlobalMetadata
; CHECK-NOT: @__spirv_BuiltInGlobalInvocationId
; CHECK-NOT: @dev_global_no_users
; CHECK-SAME: @.str
