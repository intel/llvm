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

; RUN: veczc -vecz-passes=scalarize -vecz-simd-width=4 -vecz-choices=FullScalarization -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare spir_func i64 @_Z13get_global_idj(i32)

declare <2 x float> @__vecz_b_masked_load4_Dv2_fPDv2_fDv2_b(<2 x float>*, <2 x i1>)
declare void @__vecz_b_masked_store4_Dv2_fPDv2_fDv2_b(<2 x float>, <2 x float>*, <2 x i1>)

define spir_kernel void @scalarize_masked_memops(<2 x float>* %pa, <2 x float>* %pz) {
entry:
  %idx = call spir_func i64 @_Z13get_global_idj(i32 0)
  %head = insertelement <2 x i64> undef, i64 %idx, i64 0
  %splat = shufflevector <2 x i64> %head, <2 x i64> undef, <2 x i32> zeroinitializer
  %idxs = add <2 x i64> %splat, <i64 0, i64 1>
  %mask = icmp slt <2 x i64> %idxs, <i64 8, i64 8>
  %aptr = getelementptr <2 x float>, <2 x float>* %pa, i64 %idx
  %ld = call <2 x float> @__vecz_b_masked_load4_Dv2_fPDv2_fDv2_b(<2 x float>* %aptr, <2 x i1> %mask)
  %zptr = getelementptr <2 x float>, <2 x float>* %pz, i64 %idx
  call void @__vecz_b_masked_store4_Dv2_fPDv2_fDv2_b(<2 x float> %ld, <2 x float>* %zptr, <2 x i1> %mask)
  ret void
 ; CHECK:  %idx = call spir_func i64 @_Z13get_global_idj(i32 0)
 ; CHECK:  %[[IDXS0:.*]] = add i64 %idx, 0
 ; CHECK:  %[[IDXS1:.*]] = add i64 %idx, 1
 ; CHECK:  %[[MASK0:.*]] = icmp slt i64 %[[IDXS0]], 8
 ; CHECK:  %[[MASK1:.*]] = icmp slt i64 %[[IDXS1]], 8
 ; CHECK:  %aptr = getelementptr <2 x float>, ptr %pa, i64 %idx
 ; CHECK:  %[[TMP1:.*]] = getelementptr float, ptr %aptr, i32 0
 ; CHECK:  %[[TMP2:.*]] = getelementptr float, ptr %aptr, i32 1
 ; CHECK:  %[[TMP3:.*]] = call float @__vecz_b_masked_load4_fu3ptrb(ptr %[[TMP1]], i1 %[[MASK0]])
 ; CHECK:  %[[TMP4:.*]] = call float @__vecz_b_masked_load4_fu3ptrb(ptr %[[TMP2]], i1 %[[MASK1]])
 ; CHECK:  %zptr = getelementptr <2 x float>, ptr %pz, i64 %idx
 ; CHECK:  %[[TMP6:.*]] = getelementptr float, ptr %zptr, i32 0
 ; CHECK:  %[[TMP7:.*]] = getelementptr float, ptr %zptr, i32 1
 ; CHECK:  call void @__vecz_b_masked_store4_fu3ptrb(float %[[TMP3]], ptr %[[TMP6]], i1 %[[MASK0]])
 ; CHECK:  call void @__vecz_b_masked_store4_fu3ptrb(float %[[TMP4]], ptr %[[TMP7]], i1 %[[MASK1]])
 ; CHECK:  ret void

}
