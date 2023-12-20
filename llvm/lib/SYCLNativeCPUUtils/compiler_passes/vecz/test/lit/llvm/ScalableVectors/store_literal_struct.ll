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

; Check that we do something correct when scalably packetizing struct literals.
; Right now we fail to packetize, but if we could packetize this we'd have to
; be careful as storing a struct literal containing scalable vectors is invalid
; IR.
; RUN: veczc -w 4 -vecz-scalable -vecz-passes=verify,packetizer,verify \
; RUN:   --pass-remarks-missed=vecz -S < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; CHECK: Vecz: Could not packetize  %v = load { i32, i32 }, ptr %arrayidx.p, align 4
define spir_kernel void @test_fn(ptr %p, ptr %q) {
entry:
  %idx = call i64 @__mux_get_global_id(i32 0)
  %arrayidx.p = getelementptr { i32, i32 }, ptr %p, i64 %idx
  %v = load { i32, i32 }, ptr %arrayidx.p, align 4
  %arrayidx.q = getelementptr { i32, i32 }, ptr %q, i64 %idx
  store { i32, i32 } %v, ptr %arrayidx.q, align 4
  ret void
}

declare i64 @__mux_get_global_id(i32)
