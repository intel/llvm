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

; RUN: veczc -k instrinsic -vecz-passes=packetizer -vecz-simd-width=4 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Kernels

define spir_kernel void @instrinsic(float* %in1, float* %in2, float* %in3, float* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds float, float* %in1, i64 %call
  %0 = load float, float* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, float* %in2, i64 %call
  %1 = load float, float* %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds float, float* %in3, i64 %call
  %2 = load float, float* %arrayidx2, align 4
  %3 = tail call float @llvm.fmuladd.f32(float %0, float %1, float %2)
  %arrayidx3 = getelementptr inbounds float, float* %out, i64 %call
  store float %3, float* %arrayidx3, align 4
  ret void
}

define spir_kernel void @builtin(i32* %in, i32* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds i32, i32* %in, i64 %call
  %0 = load i32, i32* %arrayidx, align 4
  %call1 = tail call spir_func i32 @_Z3absi(i32 %0)
  %arrayidx2 = getelementptr inbounds i32, i32* %out, i64 %call
  store i32 %call1, i32* %arrayidx2, align 4
  ret void
}

define spir_kernel void @user_defined(i32* %in, i32* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %add.ptr = getelementptr inbounds i32, i32* %in, i64 %call
  %add.ptr1 = getelementptr inbounds i32, i32* %out, i64 %call
  call spir_func void @defined(i32* %add.ptr, i32* %add.ptr1)
  ret void
}

define spir_kernel void @user_undefined(i32* %in, i32* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %add.ptr = getelementptr inbounds i32, i32* %in, i64 %call
  %add.ptr1 = getelementptr inbounds i32, i32* %out, i64 %call
  call spir_func void @undefined(i32* %add.ptr, i32* %add.ptr1)
  ret void
}

define spir_kernel void @cantinline(i32* %in, i32* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %add.ptr = getelementptr inbounds i32, i32* %in, i64 %call
  %add.ptr1 = getelementptr inbounds i32, i32* %out, i64 %call
  call spir_func void @dontinline(i32* %add.ptr, i32* %add.ptr1)
  ret void
}

define spir_kernel void @cantduplicate(i32* %in, i32* %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds i32, i32* %in, i64 %call
  %0 = load i32, i32* %arrayidx, align 4
  %call1 = tail call spir_func i32 @_Z3clzi(i32 %0) #1
  %arrayidx2 = getelementptr inbounds i32, i32* %out, i64 %call
  store i32 %call1, i32* %arrayidx2, align 4
  ret void
}

define spir_kernel void @optnone(i32* %in, i32* %out) #2 {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds i32, i32* %in, i64 %call
  %0 = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %out, i64 %call
  store i32 %0, i32* %arrayidx1, align 4
  ret void
}

; Declaration only functions

declare float @llvm.fmuladd.f32(float, float, float)
declare spir_func i32 @_Z3absi(i32)
declare spir_func i32 @_Z3clzi(i32) #1
declare spir_func i64 @_Z13get_global_idj(i32)
declare spir_func void @undefined(i32*, i32*)

; Functions with definitions

define spir_func void @defined(i32* %in, i32* %out) {
entry:
  %0 = load i32, i32* %in, align 4
  store i32 %0, i32* %out, align 4
  ret void
}

define spir_func void @dontinline(i32* %in, i32* %out) #0 {
entry:
  %0 = load i32, i32* %in, align 4
  store i32 %0, i32* %out, align 4
  ret void
}

; Attributes

attributes #0 = { noinline }
attributes #1 = { noduplicate }
attributes #2 = { optnone noinline }

; We should be able to handle intrinsics
; CHECK: define spir_kernel void @__vecz_v4_instrinsic
