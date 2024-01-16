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

; RUN: veczc -vecz-passes=cfg-convert,define-builtins -vecz-simd-width=4 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare i64 @__mux_get_global_id(i32)

define spir_kernel void @test_varying_if_ptr(i32 %a, ptr %b, ptr %on_true, ptr %on_false) {
entry:
  %conv = sext i32 %a to i64
  %call = call i64 @__mux_get_global_id(i32 0)
  %cmp = icmp eq i64 %conv, %call
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %idxprom = sext i32 %a to i64
  %arrayidx = getelementptr inbounds ptr, ptr %b, i64 %idxprom
  store ptr %on_true, ptr %arrayidx, align 4
  br label %if.end

if.else:
  %arrayidx2 = getelementptr inbounds ptr, ptr %b, i64 42
  store ptr %on_false, ptr %arrayidx2, align 4
  br label %if.end

if.end:
  ret void
}

; CHECK:     define void @__vecz_b_masked_store4_u3ptru3ptrb(ptr [[A:%.*]], ptr [[B:%.*]], i1 [[MASK:%.*]]) [[ATTRS:#[0-9]+]] {
; CHECK:       br i1 [[MASK]], label %[[IF:.*]], label %[[EXIT:.*]]
; CHECK:     [[IF]]:
; CHECK-NEXT:  store ptr [[A]], ptr [[B]], align 4
; CHECK-NEXT:  br label %[[EXIT]]
; CHECK:     [[EXIT]]:
; CHECK-NEXT:  ret void

; CHECK: attributes [[ATTRS]] = { norecurse nounwind }
