; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 5
; RUN: opt -passes=sycl-optimize-barriers -S < %s | FileCheck %s

; Test merging of acquire and release barriers into acquire-release.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv64-unknown-unknown"

declare spir_func void @foo()

define spir_kernel void @acq_rel_merge() {
; CHECK-LABEL: define spir_kernel void @acq_rel_merge() {
; CHECK-NEXT:    call spir_func void @foo()
; CHECK-NEXT:    call void @_Z22__spirv_ControlBarrieriii(i32 noundef 2, i32 noundef 2, i32 noundef 264)
; CHECK-NEXT:    call spir_func void @foo()
; CHECK-NEXT:    ret void
;
  call spir_func void @foo()
  call void @_Z22__spirv_ControlBarrieriii(i32 noundef 2, i32 noundef 2, i32 noundef 258)
  call void @_Z22__spirv_ControlBarrieriii(i32 noundef 2, i32 noundef 2, i32 noundef 260)
  call spir_func void @foo()
  ret void
}

declare void @_Z22__spirv_ControlBarrieriii(i32 noundef, i32 noundef, i32 noundef)
