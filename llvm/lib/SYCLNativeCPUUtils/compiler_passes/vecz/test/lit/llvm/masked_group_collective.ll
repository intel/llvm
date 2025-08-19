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

; RUN: veczc -vecz-passes="cfg-convert" -vecz-simd-width=4 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare i64 @__mux_get_local_id()
declare i32 @__mux_work_group_scan_inclusive_smax_i32(i32, i32)

; CHECK-LABEL: define spir_kernel void @__vecz_v4_foo()
; CHECK-NOT: @__vecz_b_masked___mux_work_group_scan_inclusive_smax_i32
define spir_kernel void @foo() {
entry:
  %0 = call i64 @__mux_get_local_id()
  br i1 false, label %for.body.i11, label %if.end.i105.i

for.body.i11:
  %1 = icmp slt i64 %0, 0
  br i1 %1, label %if.end.i13, label %if.end.i13

if.end.i13:
  br i1 false, label %exit, label %if.end.i105.i

if.end.i105.i:
  %2 = call i32 @__mux_work_group_scan_inclusive_smax_i32(i32 0, i32 0)
  br label %exit

exit:
  ret void
}
