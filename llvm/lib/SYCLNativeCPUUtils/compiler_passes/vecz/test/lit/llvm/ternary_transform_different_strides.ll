; Copyright (C) Codeplay Software Limited
;
; Licensed under the Apache License, Version 2.0 (the "License") with LLVM
; Exceptions; you may not use this file except in compliance with the License.
; You may obtain a copy of the License at
;
;     https://github.com/uxlfoundation/oneapi-construction-kit/blob/main/LICENSE.txt
;
; Unless required by applicable law or agreed to in writing, software
; distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
; WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
; License for the specific language governing permissions and limitations
; under the License.
;
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: veczc -k test_ternary -vecz-passes=ternary-transform -vecz-simd-width=4 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @test_ternary(i64 %a, i64 %b, i64* %c) {
entry:
  %gid = call i64 @__mux_get_global_id(i32 0)
  %gid_shift = shl i64 %gid, 1
  %cond = icmp eq i64 %a, %gid
  %c0 = getelementptr i64, i64* %c, i64 %gid
  store i64 %b, i64* %c0, align 4
  %c1 = getelementptr i64, i64* %c, i64 %gid_shift
  store i64 0, i64* %c1, align 4
  %c2 = select i1 %cond, i64* %c0, i64* %c1
  %c3 = getelementptr i64, i64* %c2, i64 %gid
  store i64 1, i64* %c3, align 4
  ret void
}

declare i64 @__mux_get_global_id(i32)

; This checks that the ternary transform is applied when the source GEPs have
; constant strides, even though they are different.

; CHECK: define spir_kernel void @__vecz_v4_test_ternary(i64 %a, i64 %b, ptr %c)
; CHECK: %gid = call i64 @__mux_get_global_id(i32 0)
; CHECK: %gid_shift = shl i64 %gid, 1
; CHECK: %cond = icmp eq i64 %a, %gid
; CHECK: %c0 = getelementptr i64, ptr %c, i64 %gid
; CHECK: store i64 %b, ptr %c0, align 4
; CHECK: %c1 = getelementptr i64, ptr %c, i64 %gid_shift
; CHECK: store i64 0, ptr %c1, align 4
; CHECK: %[[XOR:.+]] = xor i1 %cond, true
; CHECK: %[[GEP1:.+]] = getelementptr i64, ptr %c0, i64 %gid
; CHECK: %[[GEP2:.+]] = getelementptr i64, ptr %c1, i64 %gid
; CHECK: call void @__vecz_b_masked_store4_mu3ptrb(i64 1, ptr %[[GEP1]], i1 %cond)
; CHECK: call void @__vecz_b_masked_store4_mu3ptrb(i64 1, ptr %[[GEP2]], i1 %[[XOR]])
