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

; RUN: veczc -k test -vecz-simd-width=4 -vecz-passes=cfg-convert,packetizer -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @test(i32* %in, i32* %out) {
entry:
  %in.addr = alloca i32*, align 8
  %out.addr = alloca i32*, align 8
  %gid = alloca i64, align 8
  store i32* %in, i32** %in.addr, align 8
  store i32* %out, i32** %out.addr, align 8
  %call = call i64 @__mux_get_global_id(i32 0)
  store i64 %call, i64* %gid, align 8
  %0 = load i64, i64* %gid, align 8
  %rem = urem i64 %0, 16
  %cmp = icmp eq i64 %rem, 1
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %1 = load i64, i64* %gid, align 8
  %2 = load i32*, i32** %in.addr, align 8
  %arrayidx = getelementptr inbounds i32, i32* %2, i64 %1
  %3 = load i32, i32* %arrayidx, align 4
  %4 = load i64, i64* %gid, align 8
  %5 = load i32*, i32** %in.addr, align 8
  %arrayidx1 = getelementptr inbounds i32, i32* %5, i64 %4
  %call2 = call spir_func i32 bitcast (i32 (i32, i32 addrspace(1)*)* @foo to i32 (i32, i32*)*)(i32 %3, i32* %arrayidx1)
  %6 = load i64, i64* %gid, align 8
  %7 = load i32*, i32** %out.addr, align 8
  %arrayidx3 = getelementptr inbounds i32, i32* %7, i64 %6
  store i32 %call2, i32* %arrayidx3, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  %8 = load i64, i64* %gid, align 8
  %9 = load i32*, i32** %in.addr, align 8
  %arrayidx4 = getelementptr inbounds i32, i32* %9, i64 %8
  %10 = load i32, i32* %arrayidx4, align 4
  %11 = load i64, i64* %gid, align 8
  %12 = load i32*, i32** %out.addr, align 8
  %arrayidx5 = getelementptr inbounds i32, i32* %12, i64 %11
  store i32 %10, i32* %arrayidx5, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

declare i64 @__mux_get_global_id(i32)
declare spir_func i32 @foo(i32, i32 addrspace(1)*)

; CHECK: define spir_kernel void @__vecz_v4_test(
; CHECK: call spir_func i32 @__vecz_b_masked_foo(
; CHECK: call spir_func i32 @__vecz_b_masked_foo(
; CHECK: call spir_func i32 @__vecz_b_masked_foo(
; CHECK: call spir_func i32 @__vecz_b_masked_foo(
; CHECK: ret void

; CHECK: define private spir_func i32 @__vecz_b_masked_foo(i32{{( %0)?}}, ptr{{( %1)?}}, i1{{( %2)?}}
; CHECK: call spir_func i32 @foo(i32 %0, ptr %1)
