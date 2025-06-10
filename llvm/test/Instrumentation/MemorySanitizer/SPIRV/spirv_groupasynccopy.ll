; RUN: opt < %s -passes=msan -msan-instrumentation-with-call-threshold=0 -msan-eager-checks=1 -msan-poison-stack-with-call=1 -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

declare spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyiPU3AS3iPU3AS1immP13__spirv_Event(i32, ptr addrspace(3), ptr addrspace(1), i64, i64, target("spirv.Event")) nounwind
declare dso_local spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1iPU3AS3Kimm9ocl_event(i32, ptr addrspace(1), ptr addrspace(3), i64, i64, target("spirv.Event"))
declare dso_local spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS3Dv4_aPU3AS1KS_mm9ocl_event(i32, ptr addrspace(3), ptr addrspace(1), i64, i64, target("spirv.Event"))

define spir_kernel void @kernel(ptr addrspace(3) %_arg_localAcc, ptr addrspace(1) %_arg_globalAcc) sanitize_memory {
entry:
  ; CHECK: @__msan_barrier()
  ; CHECK:      [[REG1:%[0-9]+]] = ptrtoint ptr addrspace(3) %_arg_localAcc to i64
  ; CHECK-NEXT: [[REG2:%[0-9]+]] = ptrtoint ptr addrspace(1) %_arg_globalAcc to i64
  ; CHECK-NEXT: call void @__msan_unpoison_strided_copy(i64 [[REG1]], i32 3, i64 [[REG2]], i32 1, i32 4, i64 512, i64 1)
  %copy = call spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyiPU3AS3iPU3AS1immP13__spirv_Event(i32 2, ptr addrspace(3) %_arg_localAcc, ptr addrspace(1) %_arg_globalAcc, i64 512, i64 1, target("spirv.Event") zeroinitializer)

  ; CHECK: __msan_unpoison_strided_copy
  %copy2 = call spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS1iPU3AS3Kimm9ocl_event(i32 2, ptr addrspace(1) %_arg_globalAcc, ptr addrspace(3) %_arg_localAcc, i64 512, i64 1, target("spirv.Event") zeroinitializer)
  ; CHECK: __msan_unpoison_strided_copy
  %copy3 = call spir_func target("spirv.Event") @_Z22__spirv_GroupAsyncCopyjPU3AS3Dv4_aPU3AS1KS_mm9ocl_event(i32 2, ptr addrspace(3) %_arg_localAcc, ptr addrspace(1) %_arg_globalAcc, i64 512, i64 1, target("spirv.Event") zeroinitializer)
  ret void
}
