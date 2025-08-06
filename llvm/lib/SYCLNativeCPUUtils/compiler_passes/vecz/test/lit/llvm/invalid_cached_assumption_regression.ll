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

; Just check that we correctly clean up the assumption cache when vectorizing
; this function.:
; RUN: veczc -k foo -w 2 -S < %s
; RUN: not veczc -k foo -w 2 -vecz-scalable -S < %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define spir_kernel void @foo(ptr addrspace(1) nocapture readonly %_arg_v_acc) #0 {
entry:
  %v4 = tail call i64 @__mux_get_global_id(i32 0) #2
  %v5 = tail call i64 @__mux_get_global_offset(i32 0) #2
  %v6 = sub i64 %v4, %v5
  %v7 = icmp ult i64 %v6, 2147483648
  tail call void @llvm.assume(i1 %v7)
  %arrayidx.i.i = getelementptr inbounds i32, ptr addrspace(1) %_arg_v_acc, i64 %v6
  %v8 = load i32, ptr addrspace(1) %arrayidx.i.i, align 4
  ret void
}

declare void @llvm.assume(i1 noundef) #1

declare i64 @__mux_get_global_id(i32) #2
declare i64 @__mux_get_global_offset(i32) #2

attributes #0 = { convergent nounwind "mux-kernel"="entry-point" "mux-orig-fn"="foo" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn inaccessiblememonly }
attributes #2 = { alwaysinline norecurse nounwind readonly }
