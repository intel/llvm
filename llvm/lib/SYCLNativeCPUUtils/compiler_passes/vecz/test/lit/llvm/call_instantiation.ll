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

; RUN: veczc -vecz-passes=packetizer -vecz-simd-width=4 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Kernels

; We should be able to handle intrinsics
; CHECK-LABEL: define spir_kernel void @__vecz_v4_instrinsic(ptr %in1, ptr %in2, ptr %in3, ptr %out)
; CHECK: call <4 x float> @llvm.fmuladd.v4f32(<4 x float> {{%.*}}, <4 x float> {{%.*}}, <4 x float> {{%.*}})
define spir_kernel void @instrinsic(ptr %in1, ptr %in2, ptr %in3, ptr %out) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds float, ptr %in1, i64 %call
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, ptr %in2, i64 %call
  %1 = load float, ptr %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds float, ptr %in3, i64 %call
  %2 = load float, ptr %arrayidx2, align 4
  %3 = tail call float @llvm.fmuladd.f32(float %0, float %1, float %2)
  %arrayidx3 = getelementptr inbounds float, ptr %out, i64 %call
  store float %3, ptr %arrayidx3, align 4
  ret void
}

; We should be able to handle builtins for which we have a vector declaration
; in the module.
; CHECK-LABEL: define spir_kernel void @__vecz_v4_builtin(ptr %in, ptr %out)
; CHECK: = call spir_func <4 x i32> @_Z3absDv4_i(<4 x i32> {{%.*}})
define spir_kernel void @builtin(ptr %in, ptr %out) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds i32, ptr %in, i64 %call
  %0 = load i32, ptr %arrayidx, align 4
  %call1 = tail call spir_func i32 @_Z3absi(i32 %0)
  %arrayidx2 = getelementptr inbounds i32, ptr %out, i64 %call
  store i32 %call1, ptr %arrayidx2, align 4
  ret void
}

; We should be able to handle user functions for which we have a definition
; CHECK-LABEL: define spir_kernel void @__vecz_v4_user_defined(ptr %in, ptr %out)
; CHECK: call spir_func void @defined(ptr {{%.*}}, ptr {{%.*}})
; CHECK: call spir_func void @defined(ptr {{%.*}}, ptr {{%.*}})
; CHECK: call spir_func void @defined(ptr {{%.*}}, ptr {{%.*}})
; CHECK: call spir_func void @defined(ptr {{%.*}}, ptr {{%.*}})
define spir_kernel void @user_defined(ptr %in, ptr %out) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0)
  %add.ptr = getelementptr inbounds i32, ptr %in, i64 %call
  %add.ptr1 = getelementptr inbounds i32, ptr %out, i64 %call
  call spir_func void @defined(ptr %add.ptr, ptr %add.ptr1)
  ret void
}

; We should be able to handle user functions (or builtins) for which we have no
; definition
; CHECK-LABEL: define spir_kernel void @__vecz_v4_user_undefined(ptr %in, ptr %out)
; CHECK: call spir_func void @undefined(ptr {{%.*}}, ptr {{%.*}})
; CHECK: call spir_func void @undefined(ptr {{%.*}}, ptr {{%.*}})
; CHECK: call spir_func void @undefined(ptr {{%.*}}, ptr {{%.*}})
; CHECK: call spir_func void @undefined(ptr {{%.*}}, ptr {{%.*}})
define spir_kernel void @user_undefined(ptr %in, ptr %out) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0)
  %add.ptr = getelementptr inbounds i32, ptr %in, i64 %call
  %add.ptr1 = getelementptr inbounds i32, ptr %out, i64 %call
  call spir_func void @undefined(ptr %add.ptr, ptr %add.ptr1)
  ret void
}

; We should be able to handle user functions (or builtins) which we can't
; inline
; CHECK-LABEL: define spir_kernel void @__vecz_v4_cantinline(ptr %in, ptr %out)
; CHECK: call spir_func void @dontinline(ptr {{%.*}}, ptr {{%.*}})
; CHECK: call spir_func void @dontinline(ptr {{%.*}}, ptr {{%.*}})
; CHECK: call spir_func void @dontinline(ptr {{%.*}}, ptr {{%.*}})
; CHECK: call spir_func void @dontinline(ptr {{%.*}}, ptr {{%.*}})
define spir_kernel void @cantinline(ptr %in, ptr %out) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0)
  %add.ptr = getelementptr inbounds i32, ptr %in, i64 %call
  %add.ptr1 = getelementptr inbounds i32, ptr %out, i64 %call
  call spir_func void @dontinline(ptr %add.ptr, ptr %add.ptr1)
  ret void
}

; If we can't duplicate a function, we can't packetize it.
; CHECK-NOT: @__vecz_v4_cantduplicate
define spir_kernel void @cantduplicate(ptr %in, ptr %out) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds i32, ptr %in, i64 %call
  %0 = load i32, ptr %arrayidx, align 4
  %call1 = tail call spir_func i32 @_Z3clzi(i32 %0) #1
  %arrayidx2 = getelementptr inbounds i32, ptr %out, i64 %call
  store i32 %call1, ptr %arrayidx2, align 4
  ret void
}

; The optnone attribute has no impact when directly running the packetizer
; pass. The higher-level vectorization factor decisions must take this into
; account instead.
; CHECK-LABEL: define spir_kernel void @__vecz_v4_optnone(ptr %in, ptr %out)
define spir_kernel void @optnone(ptr %in, ptr %out) #2 {
entry:
  %call = call i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds i32, ptr %in, i64 %call
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %out, i64 %call
  store i32 %0, ptr %arrayidx1, align 4
  ret void
}

; Declaration only functions

declare float @llvm.fmuladd.f32(float, float, float)
declare spir_func i32 @_Z3absi(i32)
declare spir_func <4 x i32> @_Z3absDv4_i(<4 x i32>)
declare spir_func i32 @_Z3clzi(i32) #1
declare i64 @__mux_get_global_id(i32)
declare spir_func void @undefined(ptr, ptr)

; Functions with definitions

define spir_func void @defined(ptr %in, ptr %out) {
entry:
  %0 = load i32, ptr %in, align 4
  store i32 %0, ptr %out, align 4
  ret void
}

define spir_func void @dontinline(ptr %in, ptr %out) #0 {
entry:
  %0 = load i32, ptr %in, align 4
  store i32 %0, ptr %out, align 4
  ret void
}

; Attributes

attributes #0 = { noinline }
attributes #1 = { noduplicate }
attributes #2 = { optnone noinline }
