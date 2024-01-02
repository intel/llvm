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

; RUN: veczc -k test -vecz-simd-width=4 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @test(i64 %a, i64 %b, i64* %c) {
entry:
  %gid = call i64 @__mux_get_global_id(i32 0)
  %cond = icmp eq i64 %a, %gid
  %c0 = getelementptr i64, i64* %c, i64 %gid
  store i64 %b, i64* %c0, align 4
  %c1 = getelementptr i64, i64* %c, i64 0
  store i64 0, i64* %c1, align 4
  %c2 = select i1 %cond, i64* %c0, i64* %c1
  %c3 = getelementptr i64, i64* %c2, i64 %gid
  %c3.load = load i64, i64* %c3, align 4
  %c4 = getelementptr i64, i64* %c3, i64 %gid
  store i64 %c3.load, i64* %c4, align 4
  ret void
}

declare i64 @__mux_get_global_id(i32)

; Test if the scatter store is defined correctly
; CHECK: define <4 x i64> @__vecz_b_gather_load4_Dv4_mDv4_u3ptr(<4 x ptr>{{( %0)?}}) [[ATTRS:#[0-9]+]] {
; CHECK: %[[V1:[0-9]+]] = call <4 x i64> @llvm.masked.gather.v4i64.v4p0(<4 x ptr> %0, i32{{( immarg)?}} 4, <4 x i1> <i1 true, i1 true, i1 true, i1 true>,
; CHECK: ret <4 x i64> %[[V1]]

; CHECK: attributes [[ATTRS]] = { norecurse nounwind }
