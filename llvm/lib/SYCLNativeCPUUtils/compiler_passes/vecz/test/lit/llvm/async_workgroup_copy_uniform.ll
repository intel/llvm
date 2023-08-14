; Copyright (C) Codeplay Software Limited
;
; Licensed under the Apache License, Version 2.0 (the "License") with LLVM
; Exceptions; you may not use this file except in compliance with the License.
; You may obtain a copy of the License at
;
;     https://github.com/codeplaysoftware/oneapi-construction-kit/blob/main/LICENSE.txt
;
; Unless required by applicable law or agreed to in writing, software
; distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
; WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
; License for the specific language governing permissions and limitations
; under the License.
;
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: veczc -k test -vecz-simd-width=4 -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

%opencl.event_t = type opaque

; Function Attrs: nounwind
define spir_kernel void @test(i32 addrspace(1)* %input, i32 addrspace(3)* %output, i32 addrspace(1)* %elements) {
  %ev = alloca %opencl.event_t*, align 8
  %1 = call i64 @__mux_get_global_id(i32 0)
  %2 = call i64 @__mux_get_group_id(i32 0)
  %3 = call i64 @__mux_get_local_size(i32 0)
  %4 = mul i64 %3, %2
  %5 = getelementptr inbounds i32, i32 addrspace(1)* %input, i64 %4
  %6 = mul i64 %3, %2
  %7 = getelementptr inbounds i32, i32 addrspace(3)* %output, i64 %6
  %8 = getelementptr inbounds i32, i32 addrspace(1)* %elements, i64 %2
  %9 = load i32, i32 addrspace(1)* %8, align 4
  %10 = sext i32 %9 to i64
  %11 = load %opencl.event_t*, %opencl.event_t** %ev, align 8
  %12 = call spir_func %opencl.event_t* @_Z21async_work_group_copyPU3AS1iPKU3AS3im9ocl_event(i32 addrspace(1)* %5, i32 addrspace(3)* %7, i64 %10, %opencl.event_t* %11)
  %13 = trunc i64 %3 to i32
  call spir_func void @_Z17wait_group_eventsiP9ocl_event(i32 %13, %opencl.event_t** nonnull %ev)
  ret void
}

declare i64 @__mux_get_global_id(i32)
declare i64 @__mux_get_group_id(i32)
declare i64 @__mux_get_local_size(i32)
declare spir_func %opencl.event_t* @_Z21async_work_group_copyPU3AS1iPKU3AS3im9ocl_event(i32 addrspace(1)*, i32 addrspace(3)*, i64, %opencl.event_t*)
declare spir_func void @_Z17wait_group_eventsiP9ocl_event(i32, %opencl.event_t**)

; CHECK: define spir_kernel void @__vecz_v4_test

; Check if we have one and exactly one call to async_workgroup copy
; CHECK: call spir_func ptr @_Z21async_work_group_copyPU3AS1iPKU3AS3im9ocl_event
; CHECK-NOT: async_workgroup_copy

; Check if we have one and exactly one call to wait_group_events
; CHECK: call spir_func void @_Z17wait_group_eventsiP9ocl_event
; CHECK-NOT: wait_group_events
; CHECK: ret void
