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

; RUN: %veczc -k partial_linearization23 -vecz-passes=cfg-convert -S < %s | %filecheck %s

; The CFG of the following kernel is:
;
;         a
;        / \
;       /   \
;      /     \
;     b       c
;    / \     / \
;   d   e   f   g
;    \   \ /   /
;     \   X   /
;      \ / \ /
;       h   i
;        \ /
;         j
;
; * where node a is a uniform branch, and nodes b and c are varying branches.
; * where nodes d, e, f, g are divergent.
;
; With partial linearization we will have a CFG of the form:
;
;         a
;        / \
;       /   \
;      /     \
;     b       c
;    /         \
;   e - d   f - g
;        \ /
;         i
;         |
;         h
;         |
;         j
;
; The purpose of this test is to make sure we correctly handle blending in `i`
; which cannot be considered as a blend block since it is not the join point of
; either div causing blocks.
; We want to make sure the incoming blocks of the phi nodes in `i` are correctly
; translated into select instructions for the predecessors which get linearized.
;
;
;
; __kernel void partial_linearization23(__global int *out, int n) {
;   int id = get_global_id(0);
;   int ret = 0;
;
;   if (n > 10) {
;     if (id % 3 == 0) {
;       ret = n - 1; goto h;
;     } else {
;       for (int i = 0; i < n / 3; i++) { ret += 2; } goto i;
;     }
;   } else {
;     if (id % 2 == 0) {
;       ret = n * 2; goto h;
;     } else {
;       for (int i = 0; i < n + 5; i++) { ret *= 2; } goto i;
;     }
;   }
;
; h:
;   ret += 5;
;   goto end;
;
; i:
;   ret *= 10;
;   goto end;
;
; end:
;   out[id] = ret;
; }

; ModuleID = 'Unknown buffer'
source_filename = "Unknown buffer"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_kernel void @partial_linearization23(i32 addrspace(1)* %out, i32 %n) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %conv = trunc i64 %call to i32
  %cmp = icmp sgt i32 %n, 10
  br i1 %cmp, label %if.then, label %if.else7

if.then:                                          ; preds = %entry
  %rem = srem i32 %conv, 3
  %cmp2 = icmp eq i32 %rem, 0
  br i1 %cmp2, label %if.then4, label %for.cond.preheader

for.cond.preheader:                               ; preds = %if.then
  %div = sdiv i32 %n, 3
  %cmp52 = icmp sgt i32 %n, 2
  br i1 %cmp52, label %for.body.lr.ph, label %i24

for.body.lr.ph:                                   ; preds = %for.cond.preheader
  %min.iters.check = icmp ult i32 %div, 8
  br i1 %min.iters.check, label %scalar.ph, label %vector.ph

vector.ph:                                        ; preds = %for.body.lr.ph
  %n.vec = and i32 %div, -8
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %vec.phi6 = phi i32 [ 0, %vector.ph ], [ %0, %vector.body ]
  %vec.phi11 = phi i32 [ 0, %vector.ph ], [ %1, %vector.body ]
  %vec.phi17 = phi i32 [ 0, %vector.ph ], [ %2, %vector.body ]
  %vec.phi22 = phi i32 [ 0, %vector.ph ], [ %3, %vector.body ]
  %vec.phi104 = phi i32 [ 0, %vector.ph ], [ %4, %vector.body ]
  %vec.phi109 = phi i32 [ 0, %vector.ph ], [ %5, %vector.body ]
  %vec.phi1015 = phi i32 [ 0, %vector.ph ], [ %6, %vector.body ]
  %vec.phi1020 = phi i32 [ 0, %vector.ph ], [ %7, %vector.body ]
  %0 = add nuw nsw i32 %vec.phi6, 2
  %1 = add nuw nsw i32 %vec.phi11, 2
  %2 = add nuw nsw i32 %vec.phi17, 2
  %3 = add nuw nsw i32 %vec.phi22, 2
  %4 = add nuw nsw i32 %vec.phi104, 2
  %5 = add nuw nsw i32 %vec.phi109, 2
  %6 = add nuw nsw i32 %vec.phi1015, 2
  %7 = add nuw nsw i32 %vec.phi1020, 2
  %index.next = add i32 %index, 8
  %8 = icmp eq i32 %index.next, %n.vec
  br i1 %8, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  %.lcssa25 = phi i32 [ %0, %vector.body ]
  %.lcssa210 = phi i32 [ %1, %vector.body ]
  %.lcssa216 = phi i32 [ %2, %vector.body ]
  %.lcssa221 = phi i32 [ %3, %vector.body ]
  %.lcssa3 = phi i32 [ %4, %vector.body ]
  %.lcssa8 = phi i32 [ %5, %vector.body ]
  %.lcssa14 = phi i32 [ %6, %vector.body ]
  %.lcssa19 = phi i32 [ %7, %vector.body ]
  %bin.rdx7 = add nuw i32 %.lcssa3, %.lcssa25
  %bin.rdx12 = add nuw i32 %.lcssa8, %.lcssa210
  %bin.rdx18 = add nuw i32 %.lcssa14, %.lcssa216
  %bin.rdx23 = add nuw i32 %.lcssa19, %.lcssa221
  %bin.rdx1113 = add i32 %bin.rdx7, %bin.rdx12
  %bin.rdx1124 = add i32 %bin.rdx18, %bin.rdx23
  %bin.rdx1325 = add i32 %bin.rdx1113, %bin.rdx1124
  %cmp.n = icmp eq i32 %div, %n.vec
  br i1 %cmp.n, label %i24, label %scalar.ph

scalar.ph:                                        ; preds = %middle.block, %for.body.lr.ph
  %bc.resume.val = phi i32 [ %n.vec, %middle.block ], [ 0, %for.body.lr.ph ]
  %bc.merge.rdx = phi i32 [ %bin.rdx1325, %middle.block ], [ 0, %for.body.lr.ph ]
  %9 = add i32 %bc.resume.val, 1
  %10 = icmp sgt i32 %div, %9
  %smax = select i1 %10, i32 %div, i32 %9
  %11 = shl i32 %smax, 1
  %12 = shl i32 %bc.resume.val, 1
  br label %for.body

if.then4:                                         ; preds = %if.then
  %sub = add nsw i32 %n, -1
  br label %h

for.body:                                         ; preds = %for.body, %scalar.ph
  %storemerge44 = phi i32 [ %bc.resume.val, %scalar.ph ], [ %inc, %for.body ]
  %inc = add nuw nsw i32 %storemerge44, 1
  %cmp5 = icmp slt i32 %inc, %div
  br i1 %cmp5, label %for.body, label %i24.loopexit

if.else7:                                         ; preds = %entry
  %rem81 = and i32 %conv, 1
  %cmp9 = icmp eq i32 %rem81, 0
  br i1 %cmp9, label %if.then11, label %for.cond14.preheader

for.cond14.preheader:                             ; preds = %if.else7
  %add15 = add nsw i32 %n, 5
  %cmp165 = icmp sgt i32 %add15, 0
  br i1 %cmp165, label %for.body18.preheader, label %i24

for.body18.preheader:                             ; preds = %for.cond14.preheader
  %13 = add i32 %n, 5
  br label %for.body18

if.then11:                                        ; preds = %if.else7
  %mul = shl nsw i32 %n, 1
  br label %h

for.body18:                                       ; preds = %for.body18.preheader, %for.body18
  %storemerge7 = phi i32 [ %inc21, %for.body18 ], [ 0, %for.body18.preheader ]
  %ret.16 = phi i32 [ %mul19, %for.body18 ], [ 0, %for.body18.preheader ]
  %mul19 = shl nsw i32 %ret.16, 1
  %inc21 = add nuw nsw i32 %storemerge7, 1
  %exitcond = icmp ne i32 %inc21, %13
  br i1 %exitcond, label %for.body18, label %i24.loopexit1

h:                                                ; preds = %if.then11, %if.then4
  %storemerge3 = phi i32 [ %mul, %if.then11 ], [ %sub, %if.then4 ]
  %add23 = add nsw i32 %storemerge3, 5
  br label %end

i24.loopexit:                                     ; preds = %for.body
  %14 = add i32 %bc.merge.rdx, %11
  %15 = sub i32 %14, %12
  br label %i24

i24.loopexit1:                                    ; preds = %for.body18
  %mul19.lcssa = phi i32 [ %mul19, %for.body18 ]
  br label %i24

i24:                                              ; preds = %i24.loopexit1, %i24.loopexit, %for.cond14.preheader, %middle.block, %for.cond.preheader
  %ret.2 = phi i32 [ 0, %for.cond.preheader ], [ %bin.rdx1325, %middle.block ], [ 0, %for.cond14.preheader ], [ %15, %i24.loopexit ], [ %mul19.lcssa, %i24.loopexit1 ]
  %mul25 = mul nsw i32 %ret.2, 10
  br label %end

end:                                              ; preds = %i24, %h
  %storemerge2 = phi i32 [ %mul25, %i24 ], [ %add23, %h ]
  %sext = shl i64 %call, 32
  %idxprom = ashr exact i64 %sext, 32
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom
  store i32 %storemerge2, i32 addrspace(1)* %arrayidx, align 4
  ret void
}

; Function Attrs: nounwind readonly
declare spir_func i64 @_Z13get_global_idj(i32)

; CHECK: spir_kernel void @__vecz_v4_partial_linearization23
; CHECK: i24:
; CHECK: %i24.entry_mask{{.+}} = or i1
; CHECK: %i24.entry_mask{{.+}} = or i1
; CHECK: %i24.entry_mask{{.+}} = or i1
; CHECK: %i24.entry_mask{{.+}} = or i1

