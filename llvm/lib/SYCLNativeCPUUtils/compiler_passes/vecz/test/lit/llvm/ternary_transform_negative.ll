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

; RUN: %veczc -k test_negative -vecz-passes=ternary-transform -vecz-simd-width=4 -S < %s | %filecheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @test_negative(i64 %a, i64 %b, i64* %c) {
entry:
  %gid = call spir_func i64 @_Z13get_global_idj(i32 0)
  %cond = icmp eq i64 %a, %gid
  %c0 = getelementptr i64, i64* %c, i64 %gid
  %c1 = getelementptr i64, i64* %c, i64 0
  %c2 = select i1 %cond, i64* %c0, i64* %c1
  store i64 %b, i64* %c2, align 4
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)

; This checks that the ternary transform is not applied when the select is not
; accessed through an additional GEP.

; CHECK: define spir_kernel void @__vecz_v4_test_negative(i64 %a, i64 %b, ptr %c)
; CHECK: %gid = call spir_func i64 @_Z13get_global_idj(i32 0)
; CHECK: %cond = icmp eq i64 %a, %gid
; CHECK: %c0 = getelementptr i64, ptr %c, i64 %gid
; CHECK: %c1 = getelementptr i64, ptr %c, i64 0
; CHECK: %c2 = select i1 %cond, ptr %c0, ptr %c1
; CHECK: store i64 %b, ptr %c2, align 4
