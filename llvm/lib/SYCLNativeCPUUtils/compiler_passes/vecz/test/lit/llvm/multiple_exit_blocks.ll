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

; RUN: veczc -k multiple_exit_blocks -vecz-passes="function(simplifycfg,dce),mergereturn,cfg-convert" -S < %s | FileCheck %s

; ModuleID = 'Unknown buffer'
source_filename = "Unknown buffer"
target datalayout = "e-m:e-i64:64-f80:128-n8:1:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind readnone
declare i64 @__mux_get_local_id(i32)
declare i64 @__mux_get_global_id(i32)

define spir_kernel void @multiple_exit_blocks(i64 %n) {
entry:
  %gid = tail call i64 @__mux_get_global_id(i32 0)
  %lid = tail call i64 @__mux_get_local_id(i32 0)
  %cmp1 = icmp slt i64 %lid, %n
  %cmp2 = icmp slt i64 %gid, %n
  br i1 %cmp2, label %if.then, label %if.end

if.then:                                             ; preds = %entry
  %cmp3 = and i1 %cmp1, %cmp2
  br i1 %cmp3, label %if.then2, label %if.else2

if.then2:                                             ; preds = %if.then
  br label %if.else2

if.else2:                                             ; preds = %if.then, %if.then2
  br i1 %cmp1, label %if.then3, label %if.end

if.then3:                                             ; preds = %if.else2
  br label %if.end

if.end:                                             ; preds = %entry, %if.else2, %if.then3
  ret void
}

; The purpose of this test is to make sure we do not have a kernel that has more
; than one exit block after following the preparation pass.

; CHECK: define spir_kernel void @__vecz_v4_multiple_exit_blocks

; We don't want to generate any ROSCC branches:
; CHECK-NOT: entry.ROSCC:

; Only one return statement:
; CHECK: ret void
; CHECK-NOT: ret void
