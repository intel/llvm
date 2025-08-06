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

; RUN: veczc -k vecz_scalar_interleaved_load -vecz-passes=cfg-convert,packetizer -vecz-simd-width=4 -S < %s | FileCheck %s

; ModuleID = 'Unknown buffer'
source_filename = "Unknown buffer"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind readnone
declare i64 @__mux_get_global_id(i32) #0

define spir_kernel void @vecz_scalar_interleaved_load(float addrspace(1)* %out, i64 %n, float %m) {
entry:
  %gid0 = tail call i64 @__mux_get_global_id(i32 0) #0
  %gid1 = tail call i64 @__mux_get_global_id(i32 1) #0
  %cmp1 = icmp slt i64 %gid0, %n
  br i1 %cmp1, label %if.then1, label %end

if.then1:                                     ; preds = %entry
  %gep1 = getelementptr inbounds float, float addrspace(1)* %out, i64 %gid1
  %cmp2 = fcmp une float %m, 0.000000e+00
  br i1 %cmp2, label %if.then2, label %if.else2

if.then2:                                     ; preds = %if.then1
  %mul1 = mul nsw i64 %gid0, %n
  %gep2 = getelementptr inbounds float, float addrspace(1)* %gep1, i64 %mul1
  %cmp3 = icmp slt i64 %gid1, %n
  %load1 = load float, float addrspace(1)* %gep2, align 4
  %ie1 = insertelement <4 x float> undef, float %load1, i32 0
  br i1 %cmp3, label %if.then3, label %if.else3

if.then3:                                     ; preds = %if.then2
  %laod2 = load float, float addrspace(1)* %gep2, align 4
  br label %if.else3

if.else3:                                     ; preds = %if.then2, %if.then3
  %phi_load2 = phi float [ %laod2, %if.then3 ], [ 0.000000e+00, %if.then2 ]
  %ie2 = insertelement <4 x float> %ie1, float %phi_load2, i32 1
  %load3 = load float, float addrspace(1)* %gep2, align 4
  %ie3 = insertelement <4 x float> %ie2, float %load3, i32 2
  %x76 = load float, float addrspace(1)* %gep2, align 4
  %ie4 = insertelement <4 x float> %ie3, float %x76, i32 3
  br label %if.else2

if.else2:                                    ; preds = %if.else3, %if.then1
  %ret_vec = phi <4 x float> [ %ie4, %if.else3 ], [ zeroinitializer, %if.then1 ]
  %ret = extractelement <4 x float> %ret_vec, i32 0
  %ret_gep = getelementptr inbounds float, float addrspace(1)* %gep1, i64 %gid1
  store float %ret, float addrspace(1)* %ret_gep, align 4
  br label %end

end:                                    ; preds = %entry, %if.else2
  ret void
}

attributes #0 = { nounwind readnone }

; The purpose of this test is to ensure we correctly generate a scalar
; masked load for a scalar load that has a strided pointer, instead of
; generating an interleaved masked load for a non vector load (which is
; invalid).

; The middle optimizations break this test because after scalarization,
; some of the vector elements become dead code and thus, an interleaved
; load is in fact generated (although correctly, in this case)

; CHECK: spir_kernel void @__vecz_v4_vecz_scalar_interleaved_load
; CHECK: declare float @__vecz_b_masked_load4_fu3ptrU3AS1b
