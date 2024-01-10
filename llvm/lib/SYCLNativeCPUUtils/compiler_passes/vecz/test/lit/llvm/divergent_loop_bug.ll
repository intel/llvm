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

; RUN: veczc -vecz-passes=cfg-convert -S < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; CHECK: define spir_kernel void @__vecz_v4_uniform_if_then_in_divergent_block(
; CHECK-SAME:                      ptr addrspace(1) %accum_ptr, i32 %threshold, ptr addrspace(1) %out)
define spir_kernel void @uniform_if_then_in_divergent_block(ptr addrspace(1) %accum_ptr, i32 %threshold, ptr addrspace(1) %out) #4 !reqd_work_group_size !10 {
; CHECK: entry:
; CHECK: [[CMP_NOT:%.*]] = icmp slt i32 %0, %threshold
; CHECK: %cmp.not.ROSCC = icmp eq i1 [[CMP_NOT]], false
; CHECK: %cmp.not.ROSCC_any = call i1 @__vecz_b_divergence_any(i1 %cmp.not.ROSCC)
; CHECK: br i1 %cmp.not.ROSCC_any, label %entry.ROSCC, label %entry.if.end17_crit_edge
entry:
  %cosa = alloca float, align 4
  %call = tail call i64 @__mux_get_global_id(i32 0) #5
  %sext = mul i64 %call, 51539607552
  %idx.ext = ashr exact i64 %sext, 32
  %add.ptr = getelementptr inbounds i32, ptr addrspace(1) %accum_ptr, i64 %idx.ext
  %0 = load i32, ptr addrspace(1) %add.ptr, align 4
  %cmp.not = icmp slt i32 %0, %threshold
  br i1 %cmp.not, label %entry.if.end17_crit_edge, label %if.then

; CHECK: entry.ROSCC:
; CHECK: [[CMP_NOT_NOT:%.*]] = xor i1 [[CMP_NOT]], true
; CHECK: br label %if.then

entry.if.end17_crit_edge:                          ; preds = %entry
  br label %if.end17

; Ensure that only active lanes (masked by %cmp.not.not) contribute towards the
; %or.cond branch.
; CHECK: if.then:
; CHECK: call void @__vecz_b_masked_store4_fu3ptrb(float 0.000000e+00, ptr %cosa, i1 [[CMP_NOT_NOT]])
; CHECK: %1 = call spir_func float @__vecz_b_masked__Z6sincosfPf(float 0.000000e+00, ptr nonnull %cosa, i1 [[CMP_NOT_NOT]]) #9
; CHECK: %2 = call float @__vecz_b_masked_load4_fu3ptrb(ptr %cosa, i1 [[CMP_NOT_NOT]])
; CHECK: %mul7 = fmul float %2, -2.950000e+01
; CHECK: %cmp11 = fcmp uge float %mul7, 0.000000e+00
; CHECK: %cmp14 = fcmp ult float %mul7, 6.400000e+01
; CHECK: %or.cond = and i1 %cmp11, %cmp14
; CHECK: %or.cond_active = and i1 %or.cond, [[CMP_NOT_NOT]]
; CHECK: %or.cond_active_any = call i1 @__vecz_b_divergence_any(i1 %or.cond_active)
; CHECK: br i1 %or.cond_active_any, label %if.then.if.end_crit_edge, label %if.then16
if.then:                                           ; preds = %entry
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %cosa) #6
  store float 0.000000e+00, ptr %cosa, align 4
  %call4 = call spir_func float @_Z6sincosfPf(float 0.000000e+00, ptr nonnull %cosa) #7
  %1 = load float, ptr %cosa, align 4
  %mul7 = fmul float %1, -2.950000e+01
  %cmp11 = fcmp uge float %mul7, 0.000000e+00
  %cmp14 = fcmp ult float %mul7, 6.400000e+01
  %or.cond = and i1 %cmp11, %cmp14
  br i1 %or.cond, label %if.then.if.end_crit_edge, label %if.then16

if.then.if.end_crit_edge:                          ; preds = %if.then
  br label %if.end

if.then16:                                         ; preds = %if.then
  %sext2 = shl i64 %call, 32
  %idxprom = ashr exact i64 %sext2, 32
  %arrayidx = getelementptr inbounds float, ptr addrspace(1) %out, i64 %idxprom
  store float %mul7, ptr addrspace(1) %arrayidx, align 4
  br label %if.end

if.end:                                           ; preds = %if.then.if.end_crit_edge, %if.then16
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %cosa) #6
  br label %if.end17

if.end17:                                         ; preds = %entry.if.end17_crit_edge, %if.end
  ret void
}

define spir_kernel void @uniform_if_else_in_divergent_block(ptr addrspace(1) %accum_ptr, i32 %threshold, ptr addrspace(1) %out) #4 !reqd_work_group_size !10 {
; CHECK: entry:
; CHECK: [[CMP_NOT:%.*]] = icmp slt i32 %0, %threshold
; CHECK: %cmp.not.ROSCC = icmp eq i1 [[CMP_NOT]], false
; CHECK: %cmp.not.ROSCC_any = call i1 @__vecz_b_divergence_any(i1 %cmp.not.ROSCC)
; CHECK: br i1 %cmp.not.ROSCC_any, label %entry.ROSCC, label %entry.if.end17_crit_edge
entry:
  %cosa = alloca float, align 4
  %call = tail call i64 @__mux_get_global_id(i32 0) #5
  %sext = mul i64 %call, 51539607552
  %idx.ext = ashr exact i64 %sext, 32
  %add.ptr = getelementptr inbounds i32, ptr addrspace(1) %accum_ptr, i64 %idx.ext
  %0 = load i32, ptr addrspace(1) %add.ptr, align 4
  %cmp.not = icmp slt i32 %0, %threshold
  br i1 %cmp.not, label %entry.if.end17_crit_edge, label %if.then

; CHECK: entry.ROSCC:
; CHECK: [[CMP_NOT_NOT:%.*]] = xor i1 [[CMP_NOT]], true
; CHECK: br label %if.then

entry.if.end17_crit_edge:                          ; preds = %entry
  br label %if.end17

; Ensure that only active lanes (masked by %cmp.not.not) contribute towards the
; %or.cond branch.
; CHECK: if.then:
; CHECK: call void @__vecz_b_masked_store4_fu3ptrb(float 0.000000e+00, ptr %cosa, i1 [[CMP_NOT_NOT]])
; CHECK: %1 = call spir_func float @__vecz_b_masked__Z6sincosfPf(float 0.000000e+00, ptr nonnull %cosa, i1 [[CMP_NOT_NOT]]) #9
; CHECK: %2 = call float @__vecz_b_masked_load4_fu3ptrb(ptr %cosa, i1 [[CMP_NOT_NOT]])
; CHECK: %mul7 = fmul float %2, -2.950000e+01
; CHECK: %cmp11 = fcmp uge float %mul7, 0.000000e+00
; CHECK: %cmp14 = fcmp ult float %mul7, 6.400000e+01
; CHECK: %or.cond = and i1 %cmp11, %cmp14
; CHECK: %or.cond_active = and i1 %or.cond, [[CMP_NOT_NOT]]
; CHECK: %or.cond_active_any = call i1 @__vecz_b_divergence_any(i1 %or.cond_active)
; CHECK: br i1 %or.cond_active_any, label %if.else.crit_edge, label %if.then16
if.then:                                           ; preds = %entry
  call void @llvm.lifetime.start.p0(i64 4, ptr nonnull %cosa) #6
  store float 0.000000e+00, ptr %cosa, align 4
  %call4 = call spir_func float @_Z6sincosfPf(float 0.000000e+00, ptr nonnull %cosa) #7
  %1 = load float, ptr %cosa, align 4
  %mul7 = fmul float %1, -2.950000e+01
  %cmp11 = fcmp uge float %mul7, 0.000000e+00
  %cmp14 = fcmp ult float %mul7, 6.400000e+01
  %or.cond = and i1 %cmp11, %cmp14
  br i1 %or.cond, label %if.else.crit_edge, label %if.then16

if.else.crit_edge:                                 ; preds = %if.then
  br label %if.else

if.then16:                                         ; preds = %if.then
  %sext2 = shl i64 %call, 32
  %idxprom = ashr exact i64 %sext2, 32
  %arrayidx = getelementptr inbounds float, ptr addrspace(1) %out, i64 %idxprom
  store float %mul7, ptr addrspace(1) %arrayidx, align 4
  br label %if.end

if.else:
  %arrayidx2 = getelementptr inbounds float, ptr addrspace(1) %out, i64 %idxprom
  store float 1.0, ptr addrspace(1) %arrayidx2, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then16
  call void @llvm.lifetime.end.p0(i64 4, ptr nonnull %cosa) #6
  br label %if.end17

if.end17:                                         ; preds = %entry.if.end17_crit_edge, %if.end
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: nounwind
declare spir_func float @_Z6sincosfPf(float, ptr) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #1

; Function Attrs: alwaysinline norecurse nounwind memory(read)
declare i64 @__mux_get_global_id(i32) #3

attributes #0 = { norecurse nounwind "mux-kernel"="entry-point" "mux-local-mem-usage"="0" "mux-no-subgroups" "mux-orig-fn"="get_lines" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "stackrealign" "uniform-work-group-size"="true" "vecz-mode"="auto" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) "vecz-mode"="auto" }
attributes #2 = { nounwind "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "stackrealign" "vecz-mode"="auto" }
attributes #3 = { alwaysinline norecurse nounwind memory(read) "vecz-mode"="auto" }
attributes #4 = { norecurse nounwind "mux-base-fn-name"="get_lines" "mux-kernel"="entry-point" "mux-local-mem-usage"="0" "mux-no-subgroups" "mux-orig-fn"="get_lines" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "stackrealign" "uniform-work-group-size"="true" "vecz-mode"="auto" }
attributes #5 = { alwaysinline norecurse nounwind memory(read) }
attributes #6 = { nounwind }
attributes #7 = { nobuiltin nounwind "no-builtins" }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!2}
!opencl.spir.version = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 1, i32 2}
!10 = !{i32 2, i32 1, i32 1}
