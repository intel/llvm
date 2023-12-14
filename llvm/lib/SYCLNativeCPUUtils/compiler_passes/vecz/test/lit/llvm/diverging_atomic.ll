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

; RUN: veczc -w 4 -vecz-passes=cfg-convert,verify -S \
; RUN:   --pass-remarks-missed=vecz < %s 2>&1 | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-p:64:64:64-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK: Vecz: Could not apply masks for function "kernel"
; CHECK-NEXT: note: Could not apply mask to atomic instruction
; CHECK-SAME:  atomic_success = cmpxchg ptr %arrayidx.in, i32 2, i32 4 acq_rel monotonic, align 4

define spir_kernel void @kernel(ptr %in, ptr %out) {
entry:
  %gid = tail call i64 @__mux_get_global_id(i32 0)
  %cmp = icmp eq i64 %gid, 0
  br i1 %cmp, label %if.then, label %end

if.then:
  %arrayidx.in = getelementptr inbounds i32, ptr %in, i64 %gid
  %atomic_success = cmpxchg ptr %arrayidx.in, i32 2, i32 4 acq_rel monotonic, align 4
  %atomic = extractvalue { i32, i1 } %atomic_success, 0
  br label %end

end:
  %merge = phi i32 [ 0, %entry ], [ %atomic, %if.then ]
  %arrayidx.out = getelementptr inbounds i32, ptr %out, i64 %gid
  store i32 %merge, ptr %arrayidx.out, align 4
  ret void
}

declare i64 @__mux_get_global_id(i32)
