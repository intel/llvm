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

; RUN: veczc -k uniform_address_index -w 4 -S < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir-unknown-unknown"

define spir_kernel void @uniform_address_index(i32 addrspace(1)* nocapture readonly %in, i32 addrspace(1)* nocapture %out, i32 %a, i32 %b) local_unnamed_addr #0 {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0) #2
  %0 = icmp eq i32 %a, -2147483648
  %1 = icmp eq i32 %b, -1
  %2 = and i1 %0, %1
  %3 = icmp eq i32 %b, 0
  %4 = or i1 %3, %2
  %5 = select i1 %4, i32 1, i32 %b
  %div = sdiv i32 %a, %5
  %6 = trunc i64 %call to i32
  %conv1 = add i32 %div, %6
  %idxprom = sext i32 %conv1 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %idxprom
  %7 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idxprom
  store i32 %7, i32 addrspace(1)* %arrayidx3, align 4
  ret void
}

; Function Attrs: convergent nounwind readonly
declare i64 @__mux_get_global_id(i32) local_unnamed_addr #1

; It tests to ensure that the array index is correctly identified
; as having a uniform stride and generates plain vector loads and not
; gather/scatter builtin calls

; CHECK: define spir_kernel void @__vecz_v4_uniform_address_index
; CHECK: entry:
; CHECK: call i64 @__mux_get_global_id(i32 0)
; CHECK-DAG: %[[INA:.+]] = getelementptr i32, ptr addrspace(1) %in, i32 %[[X:.+]]
; CHECK-DAG: %[[LOAD:.+]] = load <4 x i32>, ptr addrspace(1) %[[INA]]
; CHECK-DAG: %[[OUTA:.+]] = getelementptr i32, ptr addrspace(1) %out, i32 %[[X:.+]]
; CHECK-DAG: store <4 x i32> %[[LOAD]], ptr addrspace(1) %[[OUTA]]
; CHECK-NOT: call <4 x i32>
