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

; RUN: %veczc -k boscc_merge2 -vecz-passes="function(simplifycfg),mergereturn,vecz-loop-rotate,cfg-convert,cleanup-divergence" -vecz-choices=LinearizeBOSCC -S < %s | %filecheck %s

; ModuleID = 'Unknown buffer'
source_filename = "kernel.opencl"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare float @llvm.fmuladd.f32(float, float, float) #2
declare void @__mux_work_group_barrier(i32, i32, i32) #3
declare spir_func float @_Z3maxff(float, float) #1
declare spir_func i64 @_Z12get_local_idj(i32) #1
declare spir_func i64 @_Z12get_group_idj(i32) #1

@fuse_conv2d_broadcast_add_relu_1_kernel0.pad_temp_shared = internal addrspace(3) global [640 x float] undef, align 4
@fuse_conv2d_broadcast_add_relu_1_kernel0.input1_shared = internal addrspace(3) global [1152 x float] undef, align 4

; Function Attrs: convergent nounwind
define spir_kernel void @boscc_merge2(float addrspace(1)* noalias %input0, float addrspace(1)* noalias %input1, float addrspace(1)* noalias %tensor, float addrspace(1)* noalias %input2) #2 {
entry:
  %compute = alloca [28 x float], align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %storemerge = phi i32 [ 0, %entry ], [ %inc2, %for.inc ]
  %cmp1 = icmp ult i32 %storemerge, 16
  br i1 %cmp1, label %if.then, label %if.else

if.then:                                      ; preds = %for.cond
  %call1 = call spir_func i64 @_Z12get_local_idj(i32 0) #5
  %call2 = call spir_func i64 @_Z12get_group_idj(i32 1) #5
  %idx1 = getelementptr inbounds [640 x float], [640 x float] addrspace(3)* @fuse_conv2d_broadcast_add_relu_1_kernel0.pad_temp_shared, i64 0, i64 %call1
  store float 0.000000e+00, float addrspace(3)* %idx1, align 4
  %cmp2 = icmp sgt i64 %call2, %call1
  br i1 %cmp2, label %if.then2, label %land.lhs.true1

land.lhs.true1:                                 ; preds = %if.then
  %call3 = call spir_func i64 @_Z12get_group_idj(i32 1) #5
  %call4 = call spir_func i64 @_Z12get_local_idj(i32 0) #5
  %cmp3 = icmp slt i64 %call3, %call4
  br i1 %cmp3, label %land.lhs.true2, label %if.then2

land.lhs.true2:                                 ; preds = %land.lhs.true1
  %call5 = call spir_func i64 @_Z12get_local_idj(i32 0) #5
  %call6 = call spir_func i64 @_Z12get_group_idj(i32 0) #5
  %cmp4 = icmp sgt i64 %call6, %call5
  br i1 %cmp4, label %if.then2, label %land.lhs.true3

land.lhs.true3:                                 ; preds = %land.lhs.true2
  %call7 = call spir_func i64 @_Z12get_group_idj(i32 0) #5
  %call8 = call spir_func i64 @_Z12get_local_idj(i32 0) #5
  %cmp5 = icmp slt i64 %call7, %call8
  br i1 %cmp5, label %cond.true4, label %if.then2

cond.true4:                                     ; preds = %land.lhs.true3
  %call9 = call spir_func i64 @_Z12get_local_idj(i32 1) #5
  %idx2 = getelementptr inbounds float, float addrspace(1)* %input0, i64 %call9
  br label %if.then2

if.then2:                                      ; preds = %cond.true4, %land.lhs.true3, %land.lhs.true2, %land.lhs.true1, %if.then
  %call10 = call spir_func i64 @_Z12get_local_idj(i32 0) #5
  %conv = trunc i64 %call10 to i32
  %idx3 = sext i32 %conv to i64
  %idx4 = getelementptr inbounds [1152 x float], [1152 x float] addrspace(3)* @fuse_conv2d_broadcast_add_relu_1_kernel0.input1_shared, i64 0, i64 %idx3
  %idx5 = getelementptr inbounds float, float addrspace(1)* %input1, i64 %idx3
  %load1 = load float, float addrspace(1)* %idx5, align 4
  store float %load1, float addrspace(3)* %idx4, align 4
  call void @__mux_work_group_barrier(i32 0, i32 1, i32 272) #4
  br label %for.cond2

for.cond2:                                     ; preds = %for.body, %if.then2
  %storemerge1 = phi i32 [ 0, %if.then2 ], [ %inc1, %for.body ]
  %cmp6 = icmp ult i32 %storemerge1, 4
  br i1 %cmp6, label %for.body, label %for.inc

for.body:                                     ; preds = %for.cond2
  %load2 = load float, float addrspace(3)* %idx4, align 4
  %fmul = call float @llvm.fmuladd.f32(float %load2, float %load2, float %load2)
  %idx6 = getelementptr inbounds [28 x float], [28 x float]* %compute, i64 0, i64 27
  store float %fmul, float* %idx6, align 4
  %inc1 = add nuw nsw i32 %storemerge1, 1
  br label %for.cond2

for.inc:                                      ; preds = %for.cond2
  %inc2 = add nuw nsw i32 %storemerge, 1
  br label %for.cond

if.else:                                      ; preds = %for.cond
  %idx7 = getelementptr inbounds [28 x float], [28 x float]* %compute, i64 0, i64 0
  %load3 = load float, float* %idx7, align 4
  %storemerge_sext = sext i32 %storemerge to i64
  %idx8 = getelementptr inbounds float, float addrspace(1)* %tensor, i64 %storemerge_sext
  store float %load3, float addrspace(1)* %idx8, align 4
  ret void
}

attributes #0 = { convergent nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nobuiltin nounwind readonly }

; CHECK: spir_kernel void @__vecz_v4_boscc_merge2
; CHECK:  br label %[[IFTHEN:.+]]

; CHECK: [[IFTHEN]]:
; CHECK: br i1 %{{.+}}, label %[[IFTHEN2:.+]], label %[[IFTHENBOSCCINDIR:.+]]

; CHECK: [[LANDLHSTRUE1UNIFORM:.+]]:
; CHECK: br i1 %{{.+}}, label %[[LANDLHSTRUE2UNIFORM:.+]], label %[[LANDLHSTRUE1UNIFORMBOSCCINDIR:.+]]

; CHECK: [[LANDLHSTRUE2UNIFORM]]:
; CHECK: br i1 %{{.+}}, label %[[IFTHEN2]], label %[[LANDLHSTRUE2UNIFORMBOSCCINDIR:.+]]

; CHECK: [[LANDLHSTRUE1UNIFORMBOSCCINDIR]]:
; CHECK: br i1 %{{.+}}, label %[[IFTHEN2]], label %[[LANDLHSTRUE2:.+]]

; CHECK: [[LANDLHSTRUE3UNIFORM:.+]]:
; CHECK: br i1 %{{.+}}, label %[[CONDTRUE4UNIFORM:.+]], label %[[LANDLHSTRUE3UNIFORMBOSCCINDIR:.+]]

; CHECK: [[CONDTRUE4UNIFORM]]:
; CHECK: br label %[[IFTHEN2]]

; CHECK: [[LANDLHSTRUE3UNIFORMBOSCCINDIR]]:
; CHECK: br i1 %{{.+}}, label %[[IFTHEN2]], label %[[CONDTRUE4:.+]]

; CHECK: [[LANDLHSTRUE1:.+]]:
; CHECK: br label %[[LANDLHSTRUE2]]

; CHECK: [[LANDLHSTRUE2]]:
; CHECK: br label %[[LANDLHSTRUE3:.+]]

; CHECK: [[LANDLHSTRUE3]]:
; CHECK: br label %[[CONDTRUE4]]

; CHECK: [[CONDTRUE4]]:
; CHECK: br label %[[IFTHEN2]]

; CHECK: [[IFTHEN2]]:
; CHECK: br label %[[FORCOND2:.+]]

; CHECK: [[LANDLHSTRUE2UNIFORMBOSCCINDIR]]:
; CHECK: br i1 %{{.+}}, label %[[LANDLHSTRUE3UNIFORM]], label %[[LANDLHSTRUE3]]

; CHECK: [[IFTHENBOSCCINDIR]]:
; CHECK: br i1 %{{.+}}, label %[[LANDLHSTRUE1UNIFORM]], label %[[LANDLHSTRUE1]]

; CHECK: [[FORCOND2]]:
; CHECK: %[[EXITCOND:.+]] = icmp
; CHECK: br i1 %[[EXITCOND]], label %[[FORBODY:.+]], label %[[FORINC:.+]]

; CHECK: [[FORBODY]]:
; CHECK: br label %[[FORCOND2]]

; CHECK: [[FORINC]]:
; CHECK: %[[EXITCOND4:.+]] = icmp
; CHECK: br i1 %[[EXITCOND4]], label %[[IFTHEN]], label %[[IFELSE:.+]]

; CHECK: [[IFELSE]]:
; CHECK: ret void
