; RUN: opt < %s -passes='function(tsan),module(tsan-module)' -tsan-instrument-func-entry-exit=0 -tsan-instrument-memintrinsics=0 -S | FileCheck %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

@dev_global = external addrspace(1) global { [4 x i32] }
@dev_global_no_users = dso_local addrspace(1) global { [4 x i32] } zeroinitializer
@.str = external addrspace(1) constant [59 x i8]
@__spirv_BuiltInGlobalInvocationId = external addrspace(1) constant <3 x i64>

; CHECK: @__TsanDeviceGlobalMetadata
; CHECK-NOT: @dev_global_no_users
; CHECK-NOT: @.str
; CHECK-NOT: @__spirv_BuiltInGlobalInvocationId
; CHECK-SAME: @dev_global

define spir_func void @test() {
entry:
  %call = call spir_func ptr addrspace(4) null(ptr addrspace(4) addrspacecast (ptr addrspace(1) @dev_global to ptr addrspace(4)), i64 0)
  ret void
}
