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

; RUN: veczc -k scalar_vector_user -vecz-simd-width=4 -vecz-passes=packetizer -vecz-choices=TargetIndependentPacketization -S < %s | FileCheck %s

; ModuleID = 'Unknown buffer'
source_filename = "Unknown buffer"
target datalayout = "e-m:e-i64:64-f80:128-n8:1:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind readnone
declare i64 @__mux_get_local_id(i32) #0

; Function Attrs: nounwind readnone
declare spir_func <4 x float> @_Z3madDv4_fS_S_(<4 x float>, <4 x float>, <4 x float>) #0

declare spir_func void @_Z7vstore4Dv4_fmPU3AS1f(<4 x float>, i64, float addrspace(1)*)

declare spir_func <4 x float> @_Z6vload4mPU3AS3Kf(i64, float addrspace(1)*)
; Function Attrs: inlinehint norecurse nounwind readnone
declare spir_func float @_Z3madfff(float, float, float) local_unnamed_addr #2

define spir_kernel void @scalar_vector_user(float addrspace(1)* %inout, i64 %n) {
entry:
  %lid = tail call i64 @__mux_get_local_id(i32 0) #0
  %inout.address = getelementptr inbounds float, float addrspace(1)* %inout, i64 %lid
  br label %loop

loop:                                              ; preds = %entry, %loop
  %madv4.prev = phi <4 x float> [ zeroinitializer, %entry ], [ %madv4, %loop ]
  %i = phi i64 [ 0, %entry ], [ %i.inc, %loop ]
  %i.inc = add nuw nsw i64 %i, 1
  %cmp = icmp slt i64 %i.inc, %n
  %inout.vload = tail call spir_func <4 x float> @_Z6vload4mPU3AS3Kf(i64 0, float addrspace(1)* %inout.address)
  %inout.vec0 = shufflevector <4 x float> %inout.vload, <4 x float> undef, <4 x i32> zeroinitializer
  %madv4 = tail call spir_func <4 x float> @_Z3madDv4_fS_S_(<4 x float> %inout.vload, <4 x float> %inout.vec0, <4 x float> %madv4.prev) #0
  br i1 %cmp, label %loop, label %end

end:                                             ; preds = %loop
  %mad.vec0 = extractelement <4 x float> %madv4, i32 0
  store float %mad.vec0, float addrspace(1)* %inout.address, align 4
  tail call spir_func void @_Z7vstore4Dv4_fmPU3AS1f(<4 x float> %madv4, i64 0, float addrspace(1)* %inout.address)
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { noduplicate }
attributes #2 = { inlinehint norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }

; The purpose of this test is to make sure we correctly scalarize an instruction
; used by both a scalar and vector instruction. We would previously try to
; scalarize its users twice thus resulting in invalid IR.

; CHECK: define spir_kernel void @__vecz_v4_scalar_vector_user
; CHECK: loop:
; CHECK: %madv4.prev{{.*}} = phi <16 x float> [ zeroinitializer, %entry ], [ %[[CONCAT:.+]], %loop ]{{$}}

; make sure the above PHI incomings are unique by looking for their definitions
; one day we might be able to super-vectorize this call, but for now we instantiate and concatenate it
; CHECK: %madv4[[S0:[0-9]+]] =
; CHECK: %madv4[[S1:[0-9]+]] =
; CHECK: %madv4[[S2:[0-9]+]] =
; CHECK: %madv4[[S3:[0-9]+]] =
; CHECK: %[[C0:.+]] = shufflevector <4 x float> %madv4[[S0]], <4 x float> %madv4[[S1]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK: %[[C1:.+]] = shufflevector <4 x float> %madv4[[S2]], <4 x float> %madv4[[S3]], <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
; CHECK: %[[CONCAT]] = shufflevector <8 x float> %[[C0]], <8 x float> %[[C1]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
