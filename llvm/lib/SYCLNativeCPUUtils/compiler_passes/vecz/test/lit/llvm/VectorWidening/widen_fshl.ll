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

; RUN: veczc -k test_calls -vecz-passes=packetizer -vecz-simd-width=16 -vecz-choices=TargetIndependentPacketization -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare i64 @__mux_get_global_id(i32)

define spir_kernel void @test_calls(i8* %pa, i8* %pb, i8* %pd) {
entry:
  %idx = call i64 @__mux_get_global_id(i32 0)
  %a = getelementptr i8, i8* %pa, i64 %idx
  %b = getelementptr i8, i8* %pb, i64 %idx
  %d = getelementptr i8, i8* %pd, i64 %idx
  %la = load i8, i8* %a, align 16
  %lb = load i8, i8* %b, align 16
  %res = tail call i8 @llvm.fshl.i8(i8 %la, i8 %lb, i8 4)
  store i8 %res, i8* %d, align 16
  ret void
}

declare i8 @llvm.fshl.i8(i8, i8, i8)

; CHECK: define spir_kernel void @__vecz_v16_test_calls(ptr %pa, ptr %pb, ptr %pd)
; CHECK: entry:

; It checks that the fshl intrinsic of i8 gets widened by a factor of 16
; CHECK: %[[LDA:.+]] = load <16 x i8>, ptr %{{.+}}
; CHECK: %[[LDB:.+]] = load <16 x i8>, ptr %{{.+}}
; CHECK: %[[RES:.+]] = call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %[[LDA]], <16 x i8> %[[LDB]], <16 x i8> {{<(i8 4(, )?)+>|splat \(i8 4\)}})
; CHECK: store <16 x i8> %[[RES]], ptr %{{.+}}

; CHECK: ret void
