; RUN: opt < %s -passes='function(tsan),module(tsan-module)' -tsan-instrument-func-entry-exit=0 -tsan-instrument-memintrinsics=0 -S | FileCheck %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

@dev_global = external addrspace(1) global { [4 x i32] }
@dev_global_no_users = dso_local addrspace(1) global { [4 x i32] } zeroinitializer
@__spirv_BuiltInGlobalInvocationId = external addrspace(1) constant <3 x i64>
@__profc_test.cpp__ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlNS0_2idILi1EEEE_clES5_ = private addrspace(1) global { ptr addrspace(1), i64 } zeroinitializer
@__profd_test.cpp__ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlNS0_2idILi1EEEE_clES5_ = private addrspace(1) global { i64, i64, i64, i64, ptr, ptr, i32, [4 x i16], i32 } { i64 6860770789959664611, i64 0, i64 sub (i64 ptrtoint (ptr addrspace(1) @__profc_test.cpp__ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlNS0_2idILi1EEEE_clES5_ to i64), i64 ptrtoint (ptr addrspace(1) @__profd_test.cpp__ZZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_ENKUlNS0_2idILi1EEEE_clES5_ to i64)), i64 0, ptr null, ptr null, i32 1, [4 x i16] zeroinitializer, i32 0 }

; CHECK: @__TsanDeviceGlobalMetadata
; CHECK-NOT: @dev_global_no_users
; CHECK-NOT: @__spirv_BuiltInGlobalInvocationId
; CHECK-NOT: @__profc
; CHECK-NOT: @__profd
; CHECK-SAME: @dev_global

define spir_func void @test() {
entry:
  %call = call spir_func ptr addrspace(4) null(ptr addrspace(4) addrspacecast (ptr addrspace(1) @dev_global to ptr addrspace(4)), i64 0)
  ret void
}
