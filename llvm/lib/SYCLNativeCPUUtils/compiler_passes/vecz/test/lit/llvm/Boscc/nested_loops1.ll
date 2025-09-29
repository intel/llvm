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

; RUN: veczc -k nested_loops1 -vecz-passes="function(simplifycfg),mergereturn,vecz-loop-rotate,cfg-convert" -vecz-choices=LinearizeBOSCC -S < %s | FileCheck %s

; ModuleID = 'Unknown buffer'
source_filename = "Unknown buffer"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind readnone
declare i64 @__mux_get_global_id(i32) #0

; Function Attrs: nounwind readnone
declare i64 @__mux_get_global_size(i32) #0

; Function Attrs: nounwind readnone
declare spir_func float @_Z3madfff(float, float, float) #0

; Function Attrs: nounwind
define spir_kernel void @nested_loops1(i32 %n, float addrspace(1)* %out) #1 {
entry:
  %gid = tail call i64 @__mux_get_global_id(i32 0) #0
  %gsize = tail call i64 @__mux_get_global_size(i32 0) #0
  %trunc_gid = trunc i64 %gid to i32
  %trunc_gsize = trunc i64 %gsize to i32
  %cmp1 = icmp slt i32 %trunc_gid, %n
  br i1 %cmp1, label %if.then1, label %end

if.then1:                                     ; preds = %16
  %cmp2 = icmp slt i32 %n, 0
  %cmp3 = icmp slt i32 %n, 0
  %cmp4 = icmp sgt i32 %n, 0
  %cmp5 = icmp slt i32 %n, 1
  br label %for.cond

for.cond:                                     ; preds = %if.else4, %if.then1
  %trunc_gid_phi = phi i32 [ %trunc_gid, %if.then1 ], [ %add3, %if.else4 ]
  %cmp6 = icmp eq i32 %trunc_gid_phi, -2147483648
  %select1 = select i1 %cmp6, i32 1, i32 %n
  %div1 = sdiv i32 %trunc_gid_phi, %select1
  br i1 %cmp2, label %if.then2, label %if.else2

if.else2:                                     ; preds = %for.cond
  %cmp7 = icmp eq i32 %n, 0
  %select2 = select i1 %cmp7, i32 1, i32 %n
  %div2 = sdiv i32 %n, %select2
  br label %if.then2

if.then2:                                     ; preds = %if.else2, %for.cond
  br i1 %cmp3, label %if.then3, label %if.else3

if.else3:                                     ; preds = %if.then2
  %cmp8 = icmp eq i32 %n, 0
  %select3 = select i1 %cmp8, i32 1, i32 %n
  %div3 = sdiv i32 %n, %select3
  br label %if.then3

if.then3:                                     ; preds = %if.else3, %if.then2
  br i1 %cmp4, label %if.then4, label %if.else4

if.then4:                                     ; preds = %if.then3
  br i1 %cmp5, label %if.else4, label %if.else5

if.else5:                                     ; preds = %if.then4
  %sext_div1 = sext i32 %div1 to i64
  %gep1 = getelementptr inbounds float, float addrspace(1)* %out, i64 %sext_div1
  %gep2 = getelementptr inbounds float, float addrspace(1)* %out, i64 %sext_div1
  br label %for.cond2

for.cond2:                                    ; preds = %if.else6, %if.else5
  %float_idx = phi float [ 0.000000e+00, %if.else5 ], [ %phi_phi_mad, %if.else6 ]
  %phi_div1_1 = phi i32 [ %div1, %if.else5 ], [ %add2, %if.else6 ]
  %i32_idx = phi i32 [ 0, %if.else5 ], [ %add2, %if.else6 ]
  %cmp9 = icmp slt i32 %phi_div1_1, %n
  br i1 %cmp9, label %if.then6, label %if.else6

if.then6:                                    ; preds = %for.cond2
  br label %for.cond3

for.cond3:                                    ; preds = %if.else7, %if.then6
  %phi_float_idx = phi float [ %float_idx, %if.then6 ], [ %phi_mad, %if.else7 ]
  %phi_div1_2 = phi i32 [ %div1, %if.then6 ], [ %add1, %if.else7 ]
  %phi_i32_idx = phi i32 [ %i32_idx, %if.then6 ], [ %add1, %if.else7 ]
  %cmp10 = icmp sgt i32 %phi_div1_2, -1
  br i1 %cmp10, label %if.then7, label %if.else7

if.then7:                                    ; preds = %for.cond3
  %sext_phi_div1_2 = sext i32 %phi_div1_2 to i64
  %gep3 = getelementptr inbounds float, float addrspace(1)* %gep1, i64 %sext_phi_div1_2
  %load1 = load float, float addrspace(1)* %gep3, align 4
  %sext_phi_i32_idx = sext i32 %phi_i32_idx to i64
  %gep4 = getelementptr inbounds float, float addrspace(1)* %gep2, i64 %sext_phi_i32_idx
  %load2 = load float, float addrspace(1)* %gep4, align 4
  %mad = tail call spir_func float @_Z3madfff(float %load1, float %load2, float %phi_float_idx) #0
  br label %if.else7

if.else7:                                    ; preds = %if.then7, %for.cond3
  %phi_mad = phi float [ %mad, %if.then7 ], [ %phi_float_idx, %for.cond3 ]
  %add1 = add nsw i32 %phi_i32_idx, %n
  %cmp11 = icmp slt i32 %add1, %div1
  br i1 %cmp11, label %for.cond3, label %if.else6

if.else6:                                    ; preds = %if.else7, %for.cond2
  %phi_phi_mad = phi float [ %float_idx, %for.cond2 ], [ %phi_mad, %if.else7 ]
  %add2 = add nsw i32 %i32_idx, %div1
  %cmp12 = icmp slt i32 %add2, %div1
  br i1 %cmp12, label %for.cond2, label %if.else4

if.else4:                                    ; preds = %if.else8, %if.then4, %if.then3
  %phi_phi_float_idx = phi float [ 0.000000e+00, %if.then3 ], [ 0.000000e+00, %if.then4 ], [ %phi_phi_mad, %if.else6 ]
  %sext_trunc_gid_phi = sext i32 %trunc_gid_phi to i64
  %gep5 = getelementptr inbounds float, float addrspace(1)* %out, i64 %sext_trunc_gid_phi
  store float %phi_phi_float_idx, float addrspace(1)* %gep5, align 4
  %add3 = add nsw i32 %trunc_gid_phi, %trunc_gsize
  %cmp13 = icmp slt i32 %add3, %n
  br i1 %cmp13, label %for.cond, label %end

end:                                    ; preds = %if.else4, %16
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }

; The purpose of this test is to make sure we correctly blend all the loops
; live through at each entry point of the divergent loops and don't create
; merge instructions for them.

; CHECK: spir_kernel void @__vecz_v4_nested_loops1
; CHECK: entry:
; CHECK: br i1 %{{.+}}, label %if.then1.uniform, label %entry.boscc_indir

; CHECK: if.then1.uniform:
; CHECK: br label %for.cond.uniform

; CHECK: entry.boscc_indir:
; CHECK: br i1 %{{.+}}, label %end, label %if.then1

; CHECK: for.cond2.uniform:
; CHECK: br i1 %{{.+}}, label %for.cond3.preheader.uniform, label %for.cond2.uniform.boscc_indir

; CHECK: for.cond2.uniform.boscc_indir:
; CHECK: br i1 %{{.+}}, label %if.else6.uniform, label %for.cond2.uniform.boscc_store

; CHECK: for.cond2.uniform.boscc_store:
; CHECK: br label %for.cond3.preheader

; CHECK: for.cond3.uniform:
; CHECK: br i1 %{{.+}}, label %if.then7.uniform, label %for.cond3.uniform.boscc_indir

; CHECK: for.cond3.uniform.boscc_indir:
; CHECK: br i1 %{{.+}}, label %if.else7.uniform, label %for.cond3.uniform.boscc_store

; CHECK: for.cond3.uniform.boscc_store:
; CHECK: br label %if.then7

; CHECK: end.loopexit.uniform:
; CHECK: br label %end

; CHECK: for.cond:
; CHECK-NOT: %{{.+}}.boscc_blend{{.+}}.merge{{.+}} =
; CHECK: br

; CHECK: for.cond2:
; CHECK-NOT: %{{.+}}.boscc_blend{{.+}}.merge{{.+}} =
; CHECK: br

; CHECK: for.cond3:
; CHECK-NOT: %{{.+}}.boscc_blend{{.+}}.merge{{.+}} =
; CHECK: br

; CHECK: if.then7:
; CHECK-NOT: %{{.+}}.boscc_blend{{.+}}.merge{{.+}} =
; CHECK: br

; CHECK: if.else4:
; CHECK-NOT: %{{.+}}.boscc_blend{{.+}}.merge{{.+}} =
; CHECK: br

; CHECK: end.loopexit:
; CHECK: br label %end

; CHECK: end:
; CHECK: ret void
