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

; RUN: veczc --vecz-passes=interleave-combine-loads -vecz-simd-width=4 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; This test checks that we can optimize interleaved accesses out of order.

define dso_local spir_kernel void @interleaved_load_4(i32 addrspace(1)* %out, i32 addrspace(1)* %in, i32 %stride) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0)
  %conv = trunc i64 %call to i32
  %call1 = tail call i64 @__mux_get_global_id(i32 1)
  %conv2 = trunc i64 %call1 to i32
  %mul = mul nsw i32 %conv2, %stride
  %add = add nsw i32 %conv, %mul
  %mul3 = shl nsw i32 %add, 1
  ; LLVM will not generate an add, but the precise form of the or instruction
  ; that gets generated depends on the LLVM version.
  ; LLVM 17-: %add4 = or i32 %mul3, 1
  ; LLVM 18+: %add4 = or disjoint i32 %mul3, 1
  ; The LLVM 17 form is not recognized as an add by LLVM 18, and the LLVM 18
  ; form uses a flag which does not exist in LLVM 17. As this is not the
  ; purpose of the test, use an add instruction here for now, and revisit this
  ; once our minimum version of LLVM is LLVM 18.
  %add4 = add nsw nuw i32 %mul3, 1
  %idxprom = sext i32 %add4 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %idxprom
  %0 = call <4 x i32> @__vecz_b_interleaved_load4_2_Dv4_jPU3AS1j(i32 addrspace(1)* %arrayidx)
  %idxprom8 = sext i32 %mul3 to i64
  %arrayidx9 = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %idxprom8
  %1 = call <4 x i32> @__vecz_b_interleaved_load4_2_Dv4_jPU3AS1j(i32 addrspace(1)* %arrayidx9)
  %sub1 = sub nsw <4 x i32> %0, %1
  %idxprom12 = sext i32 %add to i64
  %arrayidx13 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom12
  %2 = bitcast i32 addrspace(1)* %arrayidx13 to <4 x i32> addrspace(1)*
  store <4 x i32> %sub1, <4 x i32> addrspace(1)* %2, align 4
  ret void
}

; CHECK: __vecz_v4_interleaved_load_4(
; CHECK:  [[TMP1:%.*]] = load <4 x i32>, ptr addrspace(1) [[PTR:%.*]], align 4
; CHECK:  [[TMP2:%.*]] = getelementptr i32, ptr addrspace(1) [[PTR]], i32 4
; CHECK:  [[TMP4:%.*]] = load <4 x i32>, ptr addrspace(1) [[TMP2]], align 4
; CHECK:  %deinterleave = shufflevector <4 x i32> [[TMP1]], <4 x i32> [[TMP4]], <4 x i32> <i32 0, i32 2, i32 4, i32 6>
; CHECK:  %deinterleave1 = shufflevector <4 x i32> [[TMP1]], <4 x i32> [[TMP4]], <4 x i32> <i32 1, i32 3, i32 5, i32 7>
; CHECK:  %sub1 = sub nsw <4 x i32> %deinterleave1, %deinterleave


declare i64 @__mux_get_global_id(i32)
declare <4 x i32> @__vecz_b_interleaved_load4_2_Dv4_jPU3AS1j(i32 addrspace(1)*)
