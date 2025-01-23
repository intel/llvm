; RUN: opt -passes=sycl-optimize-back-to-back-barrier -S < %s | FileCheck %s
; The test checks if back-to-back __spirv_ControlBarrier and ITT annotations are
; removed.

; CHECK: define spir_func void @_Z3foov(i32 [[#Arg1:]], i32 [[#Arg2:]])
; CHECK-NEXT: call spir_func void @__itt_offload_wg_barrier_wrapper()
; CHECK-NEXT: call void @_Z22__spirv_ControlBarrieriii(i32 noundef 3, i32 noundef 1, i32 noundef 912)
; CHECK-NEXT: call spir_func void @__itt_offload_wi_resume_wrapper()
; CHECK-NEXT: call spir_func void @__itt_offload_wg_barrier_wrapper()
; CHECK-NEXT: call void @_Z22__spirv_ControlBarrieriii(i32 noundef 3, i32 noundef 2, i32 noundef 912)
; CHECK-NEXT: call spir_func void @__itt_offload_wi_resume_wrapper()
; CHECK-NEXT: call spir_func void @__itt_offload_wg_barrier_wrapper()
; CHECK-NEXT: call void @_Z22__spirv_ControlBarrieriii(i32 noundef 64, i32 noundef 2, i32 noundef 912)
; CHECK-NEXT: call spir_func void @__itt_offload_wi_resume_wrapper()
; CHECK-NEXT: call spir_func void @__itt_offload_wg_barrier_wrapper()
; CHECK-NEXT: call void @_Z22__spirv_ControlBarrieriii(i32 %0, i32 noundef 2, i32 noundef 912)
; CHECK-NEXT: call spir_func void @__itt_offload_wi_resume_wrapper()
; CHECK-NEXT: call spir_func void @__itt_offload_wg_barrier_wrapper()
; CHECK-NEXT: call void @_Z22__spirv_ControlBarrieriii(i32 %[[#Arg1]], i32 noundef 2, i32 noundef 912)
; CHECK-NEXT: call spir_func void @__itt_offload_wi_resume_wrapper()
; CHECK-NEXT: ret void

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv64-unknown-unknown"

define spir_func void @_Z3foov(i32 %0, i32 %1) {i
  call spir_func void @__itt_offload_wg_barrier_wrapper()
  call void @_Z22__spirv_ControlBarrieriii(i32 noundef 4, i32 noundef 1, i32 noundef 912)
  call spir_func void @__itt_offload_wi_resume_wrapper()

  call spir_func void @__itt_offload_wg_barrier_wrapper()
  call void @_Z22__spirv_ControlBarrieriii(i32 noundef 2, i32 noundef 1, i32 noundef 912)
  call spir_func void @__itt_offload_wi_resume_wrapper()

  call spir_func void @__itt_offload_wg_barrier_wrapper()
  call void @_Z22__spirv_ControlBarrieriii(i32 noundef 3, i32 noundef 1, i32 noundef 912)
  call spir_func void @__itt_offload_wi_resume_wrapper()

  call spir_func void @__itt_offload_wg_barrier_wrapper()
  call void @_Z22__spirv_ControlBarrieriii(i32 noundef 3, i32 noundef 1, i32 noundef 912)
  call spir_func void @__itt_offload_wi_resume_wrapper()

  call spir_func void @__itt_offload_wg_barrier_wrapper()
  call void @_Z22__spirv_ControlBarrieriii(i32 noundef 3, i32 noundef 2, i32 noundef 912)
  call spir_func void @__itt_offload_wi_resume_wrapper()

  call spir_func void @__itt_offload_wg_barrier_wrapper()
  call void @_Z22__spirv_ControlBarrieriii(i32 noundef 3, i32 noundef 2, i32 noundef 912)
  call spir_func void @__itt_offload_wi_resume_wrapper()

  call spir_func void @__itt_offload_wg_barrier_wrapper()
  call void @_Z22__spirv_ControlBarrieriii(i32 noundef 64, i32 noundef 2, i32 noundef 912)
  call spir_func void @__itt_offload_wi_resume_wrapper()

  call spir_func void @__itt_offload_wg_barrier_wrapper()
  call void @_Z22__spirv_ControlBarrieriii(i32 %0, i32 noundef 2, i32 noundef 912)
  call spir_func void @__itt_offload_wi_resume_wrapper()

  call spir_func void @__itt_offload_wg_barrier_wrapper()
  call void @_Z22__spirv_ControlBarrieriii(i32 %0, i32 noundef 2, i32 noundef 912)
  call spir_func void @__itt_offload_wi_resume_wrapper()

  call spir_func void @__itt_offload_wg_barrier_wrapper()
  call void @_Z22__spirv_ControlBarrieriii(i32 %1, i32 noundef 2, i32 noundef 912)
  call spir_func void @__itt_offload_wi_resume_wrapper()

  ret void
}

declare spir_func void @_Z22__spirv_ControlBarrieriii(i32 noundef, i32 noundef, i32 noundef)

declare spir_func void @__itt_offload_wg_barrier_wrapper()

declare spir_func void @__itt_offload_wi_resume_wrapper()
