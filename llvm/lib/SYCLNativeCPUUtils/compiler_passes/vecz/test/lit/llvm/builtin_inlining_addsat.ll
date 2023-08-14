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

; RUN: veczc -vecz-passes=builtin-inlining -vecz-simd-width=4 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @saddsatc(i8 addrspace(1)* %lhs, i8 addrspace(1)* %rhs) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds i8, i8 addrspace(1)* %lhs, i64 %call
  %0 = load i8, i8 addrspace(1)* %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds i8, i8 addrspace(1)* %rhs, i64 %call
  %1 = load i8, i8 addrspace(1)* %arrayidx1, align 1
  %call2 = tail call spir_func i8 @_Z7add_satcc(i8 %0, i8 %1)
  store i8 %call2, i8 addrspace(1)* %arrayidx1, align 1
  ret void
}

define spir_kernel void @uaddsatc(i8 addrspace(1)* %lhs, i8 addrspace(1)* %rhs) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds i8, i8 addrspace(1)* %lhs, i64 %call
  %0 = load i8, i8 addrspace(1)* %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds i8, i8 addrspace(1)* %rhs, i64 %call
  %1 = load i8, i8 addrspace(1)* %arrayidx1, align 1
  %call2 = tail call spir_func i8 @_Z7add_sathh(i8 %0, i8 %1)
  store i8 %call2, i8 addrspace(1)* %arrayidx1, align 1
  ret void
}

define spir_kernel void @saddsati(i32 addrspace(1)* %lhs, i32 addrspace(1)* %rhs) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %lhs, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %rhs, i64 %call
  %1 = load i32, i32 addrspace(1)* %arrayidx1, align 1
  %call2 = tail call spir_func i32 @_Z7add_satii(i32 %0, i32 %1)
  store i32 %call2, i32 addrspace(1)* %arrayidx1, align 1
  ret void
}

define spir_kernel void @uaddsati(i32 addrspace(1)* %lhs, i32 addrspace(1)* %rhs) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %lhs, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %rhs, i64 %call
  %1 = load i32, i32 addrspace(1)* %arrayidx1, align 1
  %call2 = tail call spir_func i32 @_Z7add_satjj(i32 %0, i32 %1)
  store i32 %call2, i32 addrspace(1)* %arrayidx1, align 1
  ret void
}

define spir_kernel void @saddsati4(<4 x i32> addrspace(1)* %lhs, <4 x i32> addrspace(1)* %rhs) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %lhs, i64 %call
  %0 = load <4 x i32>, <4 x i32> addrspace(1)* %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %rhs, i64 %call
  %1 = load <4 x i32>, <4 x i32> addrspace(1)* %arrayidx1, align 1
  %call2 = tail call spir_func <4 x i32> @_Z7add_satDv2_iS_(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %call2, <4 x i32> addrspace(1)* %arrayidx1, align 1
  ret void
}

define spir_kernel void @uaddsati4(<4 x i32> addrspace(1)* %lhs, <4 x i32> addrspace(1)* %rhs) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %lhs, i64 %call
  %0 = load <4 x i32>, <4 x i32> addrspace(1)* %arrayidx, align 1
  %arrayidx1 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %rhs, i64 %call
  %1 = load <4 x i32>, <4 x i32> addrspace(1)* %arrayidx1, align 1
  %call2 = tail call spir_func <4 x i32> @_Z7add_satDv2_jS_(<4 x i32> %0, <4 x i32> %1)
  store <4 x i32> %call2, <4 x i32> addrspace(1)* %arrayidx1, align 1
  ret void
}

declare i64 @__mux_get_global_id(i32)
declare spir_func i8 @_Z7add_satcc(i8, i8)
declare spir_func i8 @_Z7add_sathh(i8, i8)
declare spir_func i32 @_Z7add_satii(i32, i32)
declare spir_func i32 @_Z7add_satjj(i32, i32)
declare spir_func <4 x i32> @_Z7add_satDv2_iS_(<4 x i32>, <4 x i32>)
declare spir_func <4 x i32> @_Z7add_satDv2_jS_(<4 x i32>, <4 x i32>)

; CHECK: define spir_kernel void @__vecz_v4_saddsatc(
; CHECK: = call i8 @llvm.sadd.sat.i8(i8 %{{.*}}, i8 %{{.*}})

; CHECK: define spir_kernel void @__vecz_v4_uaddsatc(
; CHECK: = call i8 @llvm.uadd.sat.i8(i8 %{{.*}}, i8 %{{.*}})

; CHECK: define spir_kernel void @__vecz_v4_saddsati(
; CHECK: = call i32 @llvm.sadd.sat.i32(i32 %{{.*}}, i32 %{{.*}})

; CHECK: define spir_kernel void @__vecz_v4_uaddsati(
; CHECK: = call i32 @llvm.uadd.sat.i32(i32 %{{.*}}, i32 %{{.*}})

; CHECK: define spir_kernel void @__vecz_v4_saddsati4(
; CHECK: = call <4 x i32> @llvm.sadd.sat.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})

; CHECK: define spir_kernel void @__vecz_v4_uaddsati4(
; CHECK: = call <4 x i32> @llvm.uadd.sat.v4i32(<4 x i32> %{{.*}}, <4 x i32> %{{.*}})
