; RUN: opt < %s -passes='function(tsan),module(tsan-module)' -tsan-instrument-memintrinsics=0 -S | FileCheck %s
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

define spir_kernel void @CheckDeviceBarrier() {
; CHECK-LABEL: void @CheckDeviceBarrier
entry:
  call spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef 1, i32 noundef 1, i32 noundef 912)
  ; CHECK: call void @__tsan_device_barrier
  br label %exit

exit: ; preds = %entry
  ret void
}

define spir_kernel void @CheckGroupBarrier() {
; CHECK-LABEL: void @CheckGroupBarrier
entry:
  call spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef 2, i32 noundef 2, i32 noundef 912)
  ; CHECK: call void @__tsan_group_barrier
  br label %exit

exit: ; preds = %entry
  ret void
}

declare spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef, i32 noundef, i32 noundef)

