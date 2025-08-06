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

; RUN: %pp-llvm-ver -o %t < %s --llvm-ver %LLVMVER
; RUN: veczc -k irreducible_loop -S < %s | FileCheck %t

; ModuleID = 'Unknown buffer'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_kernel void @irreducible_loop(i32 addrspace(1)* %src, i32 addrspace(1)* %dst) #0 {
entry:
  %call = call i64 @__mux_get_global_id(i32 0) #2
  %arrayidx4 = getelementptr inbounds i32, i32 addrspace(1)* %dst, i64 %call
  %ld = load i32, i32 addrspace(1)* %arrayidx4, align 4
  %cmp = icmp sgt i32 %ld, -1
  br i1 %cmp, label %label, label %do.body

do.body:                                          ; preds = %entry, %label
  %id.0 = phi i64 [ %conv10, %label ], [ %call, %entry ]
  br label %label

label:                                            ; preds = %entry, %do.body
  %id.1 = phi i64 [ %id.0, %do.body ], [ %call, %entry ]
  %conv10 = add i64 %id.1, 1
  %cmp11 = icmp slt i64 %conv10, 16
  br i1 %cmp11, label %do.body, label %do.end

do.end:                                           ; preds = %label
  ret void
}

; Function Attrs: convergent nounwind readonly
declare i64 @__mux_get_global_id(i32)

; CHECK: define spir_kernel void @__vecz_v4_irreducible_loop
; CHECK: entry:
; CHECK-LT20:   br label %irr.guard.outer

; CHECK-LT20: irr.guard.outer:                                  ; preds = %irr.guard.pure_exit, %entry
; CHECK:   br label %irr.guard

; CHECK-LT20: do.end:                                           ; preds = %irr.guard.pure_exit
; CHECK-LT20:   ret void

; CHECK: irr.guard:
; CHECK:   br i1 %{{.+}}, label %irr.guard.pure_exit, label %irr.guard

; CHECK: irr.guard.pure_exit:                              ; preds = %irr.guard
; CHECK-LT20:   br i1 %{{.+}}, label %do.end, label %irr.guard.outer
; CHECK-GE20:   ret void
