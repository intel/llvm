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

; RUN: veczc -k test -vecz-choices=InstantiateCallsInLoops -vecz-simd-width=4 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

@.str = private unnamed_addr addrspace(2) constant [23 x i8] c"Hello from %d with %d\0A\00", align 1
@.str.1 = private unnamed_addr addrspace(2) constant [14 x i8] c"Hello from %d\00", align 1

define spir_kernel void @test(i32 addrspace(1)* %in) {
entry:
  %call = call i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call1 = call spir_func i32 (i8 addrspace(2)*, ...) @printf(i8 addrspace(2)* getelementptr inbounds ([23 x i8], [23 x i8] addrspace(2)* @.str, i64 0, i64 0), i64 %call, i32 %0)
  %call2 = call spir_func i32 (i8 addrspace(2)*, ...) @printf(i8 addrspace(2)* getelementptr inbounds ([14 x i8], [14 x i8] addrspace(2)* @.str.1, i64 0, i64 0), i64 %call)
  ret void
}

declare i64 @__mux_get_global_id(i32)
declare extern_weak spir_func i32 @printf(i8 addrspace(2)*, ...)

; CHECK: define spir_kernel void @__vecz_v4_test(ptr addrspace(1) %in)

; CHECK: [[LOOPHEADER1:instloop.header.*]]:
; CHECK: %[[INSTANCE1:instance.*]] = phi i32 [ 0, {{.+}} ], [ %[[V7:[0-9]+]], %[[LOOPBODY1:instloop.body.*]] ]
; CHECK: %[[V3:[0-9]+]] = icmp ult i32 %[[INSTANCE1]], 4
; CHECK: br i1 %[[V3]], label %[[LOOPBODY1]], label {{.+}}

; CHECK: [[LOOPBODY1]]:
; CHECK: %[[V4:[0-9]+]] = extractelement <4 x i64> %0, i32 %[[INSTANCE1]]
; CHECK: %[[V5:[0-9]+]] = extractelement <4 x i32> %{{.+}}, i32 %[[INSTANCE1]]
; CHECK: call spir_func i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) @{{.+}}, i64 %[[V4]], i32 %[[V5]])
; CHECK: %[[V7]] = add {{(nuw |nsw )*}}i32 %[[INSTANCE1]], 1
; CHECK: br label %[[LOOPHEADER1]]

; CHECK: [[LOOPHEADER2:instloop.header.*]]:
; CHECK: %[[INSTANCE3:.+]] = phi i32 [ %[[V11:[0-9]+]], %[[LOOPBODY2:instloop.body.*]] ], [ 0, {{.+}} ]
; CHECK: %[[V8:[0-9]+]] = icmp ult i32 %[[INSTANCE3]], 4
; CHECK: br i1 %[[V8]], label %[[LOOPBODY2]], label {{.+}}

; CHECK: [[LOOPBODY2]]:
; CHECK: %[[V9:[0-9]+]] = extractelement <4 x i64> %0, i32 %[[INSTANCE3]]
; CHECK: call spir_func i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) @{{.+}}, i64 %[[V9]])
; CHECK: %[[V11]] = add {{(nuw |nsw )*}}i32 %[[INSTANCE3]], 1
; CHECK: br label %[[LOOPHEADER2]]

; CHECK: ret void
