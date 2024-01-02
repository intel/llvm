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

; REQUIRES: llvm-13+
; RUN: veczc -vecz-scalable -vecz-simd-width=4 -vecz-choices=VectorPredication -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare i64 @__mux_get_global_id(i32)

; With VP all gathers become masked ones.
define spir_kernel void @unmasked_gather(i32 addrspace(1)* %a, i32 addrspace(1)* %b) {
entry:
  %call = call i64 @__mux_get_global_id(i32 0)
  %rem = urem i64 %call, 3
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %a, i64 %rem
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %b, i64 %call
  store i32 %0, i32 addrspace(1)* %arrayidx3, align 4
  ret void
}

; CHECK: define spir_kernel void @__vecz_nxv4_vp_unmasked_gather(
; CHECK: [[v:%.*]] = call <vscale x 4 x i32> @__vecz_b_masked_gather_load4_vp_u5nxv4ju14nxv4u3ptrU3AS1u5nxv4bj(<vscale x 4 x ptr addrspace(1)> %{{.*}})
; CHECK: call void @llvm.vp.store.nxv4i32.p1(<vscale x 4 x i32> [[v]],


; With VP all scatters become masked ones.
define spir_kernel void @unmasked_scatter(i32 addrspace(1)* %a, i32 addrspace(1)* %b) {
entry:
  %call = call i64 @__mux_get_global_id(i32 0)
  %rem = urem i64 %call, 3
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %a, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %b, i64 %rem
  store i32 %0, i32 addrspace(1)* %arrayidx3, align 4
  ret void
}

; CHECK: define spir_kernel void @__vecz_nxv4_vp_unmasked_scatter(
; CHECK: [[v:%.*]] = call <vscale x 4 x i32> @llvm.vp.load.nxv4i32.p1(
; CHECK: call void @__vecz_b_masked_scatter_store4_vp_u5nxv4ju14nxv4u3ptrU3AS1u5nxv4bj(<vscale x 4 x i32> [[v]],

; CHECK: define <vscale x 4 x i32> @__vecz_b_masked_gather_load4_vp_u5nxv4ju14nxv4u3ptrU3AS1u5nxv4bj(<vscale x 4 x ptr addrspace(1)> %0, <vscale x 4 x i1> %1, i32 %2) [[ATTRS:#[0-9]+]] {
; CHECK:   %3 = call <vscale x 4 x i32> @llvm.vp.gather.nxv4i32.nxv4p1(<vscale x 4 x ptr addrspace(1)> %0, <vscale x 4 x i1> %1, i32 %2)
; CHECK:   ret <vscale x 4 x i32> %3

; CHECK: define void @__vecz_b_masked_scatter_store4_vp_u5nxv4ju14nxv4u3ptrU3AS1u5nxv4bj(<vscale x 4 x i32> %0, <vscale x 4 x ptr addrspace(1)> %1, <vscale x 4 x i1> %2, i32 %3) [[ATTRS]] {
; CHECK: entry:
; CHECK:   call void @llvm.vp.scatter.nxv4i32.nxv4p1(<vscale x 4 x i32> %0, <vscale x 4 x ptr addrspace(1)> %1, <vscale x 4 x i1> %2, i32 %3)
; CHECK:   ret void
