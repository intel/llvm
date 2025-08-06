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

; RUN: veczc -k boscc_merge3 -vecz-passes="function(instcombine,simplifycfg),mergereturn,vecz-loop-rotate,function(loop(indvars)),cfg-convert,cleanup-divergence" -vecz-choices=LinearizeBOSCC -S < %s | FileCheck %s

; ModuleID = 'Unknown buffer'
source_filename = "Unknown buffer"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind readnone
declare i64 @__mux_get_global_id(i32) #0

; Function Attrs: nounwind readnone
declare spir_func <4 x float> @_Z6vload4mPU3AS1Kf(i64, float addrspace(1)*)

define spir_kernel void @boscc_merge3(float addrspace(1)* %out, i64 noundef %n, float noundef %m) {
entry:
  %gid0 = tail call i64 @__mux_get_global_id(i32 0) #0
  %gid1 = tail call i64 @__mux_get_global_id(i32 1) #0
  %cmp1 = icmp slt i64 %gid0, %n
  br i1 %cmp1, label %if.then1, label %end

if.then1:                                     ; preds = %entry
  %gep1 = getelementptr inbounds float, float addrspace(1)* %out, i64 %gid1
  %cmp2 = fcmp une float %m, 0.000000e+00
  br i1 %cmp2, label %if.then2, label %if.end1

if.then2:                                     ; preds = %if.then1
  %cmp3 = icmp sge i64 %gid1, %n
  %gep2 = getelementptr inbounds float, float addrspace(1)* %gep1, i64 %gid0
  br i1 %cmp3, label %if.then3, label %if.else3

if.then3:                                     ; preds = %x51
  %load1 = load float, float addrspace(1)* %gep2, align 4
  %ie_load1 = insertelement <4 x float> undef, float %load1, i32 0
  br label %if.end2

if.else3:                                    ; preds = %x51
  %vload1 = tail call spir_func <4 x float> @_Z6vload4mPU3AS1Kf(i64 0, float addrspace(1)* %gep2)
  %cmp4 = icmp slt i64 %gid0, %n
  br i1 %cmp4, label %if.then4, label %if.end2

if.then4:                                    ; preds = %x175
  %vload2 = tail call spir_func <4 x float> @_Z6vload4mPU3AS1Kf(i64 4, float addrspace(1)* %gep2)
  br label %if.end2

if.end2:                                    ; preds = %x274, %x271, %if.then4, %x175, %x155, %x132
  %phi_gep2_load = phi <4 x float> [ %ie_load1, %if.then3 ], [ %vload2, %if.then4 ], [ %vload1, %if.else3 ]
  %ie_m = insertelement <4 x float> undef, float %m, i32 0
  %shuffle_ie_m = shufflevector <4 x float> %ie_m, <4 x float> undef, <4 x i32> zeroinitializer
  %fmul = fmul <4 x float> %shuffle_ie_m, %phi_gep2_load
  br label %if.end1

if.end1:                                    ; preds = %if.end2, %if.then1
  %phi_fmul = phi <4 x float> [ %fmul, %if.end2 ], [ zeroinitializer, %if.then1 ]
  %ee0 = extractelement <4 x float> %phi_fmul, i32 0
  store float %ee0, float addrspace(1)* %gep1, align 4
  br label %end

end:
  ret void
}

attributes #0 = { nounwind readnone }

; CHECK: spir_kernel void @__vecz_v4_boscc_merge3
; CHECK: entry:
; CHECK: %[[BOSCC:.+]] = call i1 @__vecz_b_divergence_all(i1 %cmp1)
; CHECK: br i1 %[[BOSCC]], label %if.then1.uniform, label %entry.boscc_indir

; CHECK: if.then1.uniform:
; CHECK: %gep1.uniform =
; CHECK: br i1 %cmp2.uniform, label %if.then2.uniform, label %if.end1.uniform

; CHECK: if.else3.uniform:
; CHECK: %[[BOSCC2:.+]] = call i1 @__vecz_b_divergence_all(i1 %{{if.then4.uniform.exit_mask|cmp4.uniform}})
; CHECK: br i1 %[[BOSCC2]], label %if.then4.uniform, label %if.else3.uniform.boscc_indir

; CHECK: if.else3.uniform.boscc_indir:
; CHECK: %[[BOSCC3:.+]] = call i1 @__vecz_b_divergence_all(i1 %if.end2.uniform.exit_mask)
; CHECK: br i1 %[[BOSCC3]], label %if.end2.uniform, label %if.then4

; CHECK: if.then1:
; CHECK: %gep1 =
; CHECK: br i1 %cmp2, label %if.then2, label %if.end1

; Generalizing the expected %cmp3 value because the 'icmp' could go off
; by one BB between LLVM versions. Therefore we can get %cmp3.not.
; CHECK: if.then2:
; CHECK: br i1 %cmp3{{(.+)?}}, label %if.else3, label %if.then3

; CHECK: if.then3:
; CHECK: br label %if.end2

; CHECK: if.else3:
; CHECK: br label %if.then4

; CHECK: if.then4:
; CHECK: %gep1.boscc_blend = phi ptr addrspace(1) [ %gep1.uniform, %if.else3.uniform.boscc_indir ], [ %gep1, %if.else3 ]
; CHECK: br label %if.end2

; CHECK: if.end2:

; Check we have correctly blended the instruction during the BOSCC connection
; rather than while repairing the SSA form.
; CHECK-NOT: %gep1.boscc_blend.merge{{.*}} = phi
; CHECK: %gep1.boscc_blend{{[0-9]*}} = phi ptr addrspace(1) [ %gep1.boscc_blend{{[0-9]*}}, %if.then4 ], [ %gep1, %if.then3 ]
; CHECK: br label %if.end1

; CHECK: if.end1:

; Check we have correctly blended the instruction during the BOSCC connection
; rather than while repairing the SSA form.
; CHECK-NOT: %gep1.boscc_blend.merge{{.*}} = phi
; CHECK: %gep1.boscc_blend{{[0-9]*}} = phi ptr addrspace(1) [ %gep1.boscc_blend{{[0-9]*}}, %if.end2 ], [ %gep1, %if.then1 ]
; CHECK: br label %end
