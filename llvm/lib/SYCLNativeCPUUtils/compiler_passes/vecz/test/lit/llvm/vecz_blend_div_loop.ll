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

; RUN: %veczc -k blend_div_loop -vecz-passes="function(simplifycfg),mergereturn,vecz-loop-rotate,cfg-convert,cleanup-divergence" -S < %s | %filecheck %s

; ModuleID = 'Unknown buffer'
source_filename = "kernel.opencl"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_kernel void @blend_div_loop(i8 addrspace(1)* %src1ptr, i32 %src1_step, i32 %src1_offset, i8 addrspace(1)* %dstptr, i32 %dst_step, i32 %dst_offset, i32 %dst_rows, i32 %dst_cols, i8 addrspace(1)* %src2ptr, i32 %src2_step, i32 %src2_offset, i8 addrspace(1)* %src3ptr, i32 %src3_step, i32 %src3_offset, i32 %rowsPerWI) #0 {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0) #2
  %conv = trunc i64 %call to i32
  %call1 = call spir_func i64 @_Z13get_global_idj(i32 1) #2
  %0 = trunc i64 %call1 to i32
  %conv3 = mul i32 %0, %rowsPerWI
  %cmp = icmp slt i32 %conv, %dst_cols
  br i1 %cmp, label %if.then, label %if.end62

if.then:                                          ; preds = %entry
  %call5 = call spir_func i32 @_Z5mad24iii(i32 %conv, i32 1, i32 %src1_offset) #2
  %call6 = call spir_func i32 @_Z5mad24iii(i32 %conv3, i32 %src1_step, i32 %call5) #2
  %call7 = call spir_func i32 @_Z5mad24iii(i32 %conv, i32 1, i32 %dst_offset) #2
  %call8 = call spir_func i32 @_Z5mad24iii(i32 %conv3, i32 %dst_step, i32 %call7) #2
  %call9 = call spir_func i32 @_Z5mad24iii(i32 %conv, i32 1, i32 %src2_offset) #2
  %call10 = call spir_func i32 @_Z5mad24iii(i32 %conv3, i32 %src2_step, i32 %call9) #2
  %call11 = call spir_func i32 @_Z5mad24iii(i32 %conv, i32 1, i32 %src3_offset) #2
  %call12 = call spir_func i32 @_Z5mad24iii(i32 %conv3, i32 %src3_step, i32 %call11) #2
  %add = add nsw i32 %conv3, %rowsPerWI
  %call13 = call spir_func i32 @_Z3minii(i32 %dst_rows, i32 %add) #2
  br label %for.cond

for.cond:                                         ; preds = %for.end54, %if.then
  %src1_index.0 = phi i32 [ %call6, %if.then ], [ %add59, %for.end54 ]
  %dst_index.0 = phi i32 [ %call8, %if.then ], [ %add60, %for.end54 ]
  %src2_index.0 = phi i32 [ %call10, %if.then ], [ %add55, %for.end54 ]
  %src3_index.0 = phi i32 [ %call12, %if.then ], [ %add56, %for.end54 ]
  %y.0 = phi i32 [ %conv3, %if.then ], [ %inc58, %for.end54 ]
  %cmp14 = icmp slt i32 %y.0, %call13
  br i1 %cmp14, label %for.body, label %if.end62

for.body:                                         ; preds = %for.cond
  %idx.ext = sext i32 %src1_index.0 to i64
  %add.ptr = getelementptr inbounds i8, i8 addrspace(1)* %src1ptr, i64 %idx.ext
  %idx.ext16 = sext i32 %dst_index.0 to i64
  %add.ptr17 = getelementptr inbounds i8, i8 addrspace(1)* %dstptr, i64 %idx.ext16
  %idx.ext18 = sext i32 %src2_index.0 to i64
  %add.ptr19 = getelementptr inbounds i8, i8 addrspace(1)* %src2ptr, i64 %idx.ext18
  %idx.ext20 = sext i32 %src3_index.0 to i64
  %add.ptr21 = getelementptr inbounds i8, i8 addrspace(1)* %src3ptr, i64 %idx.ext20
  br label %for.cond22

for.cond22:                                       ; preds = %for.inc49, %for.body
  %src1.0 = phi i8 addrspace(1)* [ %add.ptr, %for.body ], [ %add.ptr51, %for.inc49 ]
  %src2.0 = phi i8 addrspace(1)* [ %add.ptr19, %for.body ], [ %add.ptr52, %for.inc49 ]
  %src3.0 = phi i8 addrspace(1)* [ %add.ptr21, %for.body ], [ %add.ptr53, %for.inc49 ]
  %px.0 = phi i32 [ 0, %for.body ], [ %inc50, %for.inc49 ]
  %cmp23 = icmp eq i32 %px.0, 0
  br i1 %cmp23, label %for.body25, label %for.end54

for.body25:                                       ; preds = %for.cond22
  %1 = zext i32 %px.0 to i64
  %arrayidx = getelementptr inbounds i8, i8 addrspace(1)* %add.ptr17, i64 %1
  store i8 -1, i8 addrspace(1)* %arrayidx, align 1
  br label %for.cond26

for.cond26:                                       ; preds = %for.inc, %for.body25
  %storemerge = phi i32 [ 0, %for.body25 ], [ %inc, %for.inc ]
  %cmp27 = icmp eq i32 %storemerge, 0
  br i1 %cmp27, label %for.body29, label %for.inc49

for.body29:                                       ; preds = %for.cond26
  %2 = zext i32 %storemerge to i64
  %arrayidx31 = getelementptr inbounds i8, i8 addrspace(1)* %src2.0, i64 %2
  %3 = load i8, i8 addrspace(1)* %arrayidx31, align 1
  %4 = zext i32 %storemerge to i64
  %arrayidx34 = getelementptr inbounds i8, i8 addrspace(1)* %src1.0, i64 %4
  %5 = load i8, i8 addrspace(1)* %arrayidx34, align 1
  %cmp36 = icmp ugt i8 %3, %5
  br i1 %cmp36, label %if.then46, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %for.body29
  %6 = zext i32 %storemerge to i64
  %arrayidx39 = getelementptr inbounds i8, i8 addrspace(1)* %src3.0, i64 %6
  %7 = load i8, i8 addrspace(1)* %arrayidx39, align 1
  %8 = zext i32 %storemerge to i64
  %arrayidx42 = getelementptr inbounds i8, i8 addrspace(1)* %src1.0, i64 %8
  %9 = load i8, i8 addrspace(1)* %arrayidx42, align 1
  %cmp44 = icmp ult i8 %7, %9
  br i1 %cmp44, label %if.then46, label %for.inc

if.then46:                                        ; preds = %lor.lhs.false, %for.body29
  %10 = zext i32 %px.0 to i64
  %arrayidx48 = getelementptr inbounds i8, i8 addrspace(1)* %add.ptr17, i64 %10
  store i8 0, i8 addrspace(1)* %arrayidx48, align 1
  br label %for.inc49

for.inc:                                          ; preds = %lor.lhs.false
  %inc = add nuw nsw i32 %storemerge, 1
  br label %for.cond26

for.inc49:                                        ; preds = %if.then46, %for.cond26
  %inc50 = add nuw nsw i32 %px.0, 1
  %add.ptr51 = getelementptr inbounds i8, i8 addrspace(1)* %src1.0, i64 1
  %add.ptr52 = getelementptr inbounds i8, i8 addrspace(1)* %src2.0, i64 1
  %add.ptr53 = getelementptr inbounds i8, i8 addrspace(1)* %src3.0, i64 1
  br label %for.cond22

for.end54:                                        ; preds = %for.cond22
  %add55 = add nsw i32 %src2_index.0, %src2_step
  %add56 = add nsw i32 %src3_index.0, %src3_step
  %inc58 = add nsw i32 %y.0, 1
  %add59 = add nsw i32 %src1_index.0, %src1_step
  %add60 = add nsw i32 %dst_index.0, %dst_step
  br label %for.cond

if.end62:                                         ; preds = %for.cond, %entry
  ret void
}

; Function Attrs: convergent nounwind readonly
declare spir_func i64 @_Z13get_global_idj(i32) #1

; Function Attrs: convergent nounwind readonly
declare spir_func i32 @_Z5mad24iii(i32, i32, i32) #1

; Function Attrs: convergent nounwind readonly
declare spir_func i32 @_Z3minii(i32, i32) #1

attributes #0 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nobuiltin nounwind readonly }

; The purpose of this test is to make sure we correctly replace the uses of
; divergent loop update masks outside the loop, even in the pure exit.

; CHECK: spir_kernel void @__vecz_v4_blend_div_loop
; CHECK: for.cond26.pure_exit:
; CHECK: %if.then46.entry_mask{{[0-9]+}} = or i1 %if.then46.loop_exit_mask{{[0-9]+}}.blend, %if.then46.loop_exit_mask.blend
