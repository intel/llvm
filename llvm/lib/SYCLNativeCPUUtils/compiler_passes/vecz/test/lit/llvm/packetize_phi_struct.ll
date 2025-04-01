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

; CHECK-LABEL: define spir_kernel void @__vecz_v4_foo()
define spir_kernel void @foo() {
; CHECK-LABEL: entry:
entry:
  ; CHECK: %0 = call { <4 x i64>, <4 x i1> } @__vecz_b_v4_masked_cmpxchg_align8_monotonic_monotonic_1_Dv4_u3ptrDv4_mDv4_mDv4_b(
  %0 = cmpxchg ptr null, i64 0, i64 0 monotonic monotonic, align 8
  ; CHECK: br label %bb.1
  br label %bb.1

; CHECK-LABEL: bb.1:
bb.1:
  ; CHECK: %1 = phi { <4 x i64>, <4 x i1> } [ %0, %bb.1 ], [ %0, %entry ]
  %1 = phi { i64, i1 } [ %0, %bb.1 ], [ %0, %entry ]
  ; CHECK: %2 = extractvalue { <4 x i64>, <4 x i1> } %1, 0
  %2 = extractvalue { i64, i1 } %1, 0
  ; %3 = call { <4 x i64>, <4 x i1> } @__vecz_b_v4_masked_cmpxchg_align8_monotonic_monotonic_1_Dv4_u3ptrDv4_mDv4_mDv4_b(
  %3 = cmpxchg ptr null, i64 0, i64 %2 monotonic monotonic, align 8
  ; CHECK: br label %bb.1
  br label %bb.1
}
