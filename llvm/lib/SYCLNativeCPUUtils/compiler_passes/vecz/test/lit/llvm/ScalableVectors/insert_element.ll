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
; RUN: veczc -k insert_element -vecz-scalable -vecz-simd-width=4 -S < %s | FileCheck %s --check-prefix=IE
; RUN: veczc -k insert_element_uniform -vecz-scalable -vecz-simd-width=4 -S < %s | FileCheck %s --check-prefix=IE-UNI
; RUN: veczc -k insert_element_varying_indices -vecz-scalable -vecz-simd-width=4 -S < %s | FileCheck %s --check-prefix=IE-INDICES
; RUN: veczc -k insert_element_bool -vecz-scalable -vecz-simd-width=4 -S < %s | FileCheck %s --check-prefix=IE-BOOL

target triple = "spir64-unknown-unknown"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare i64 @__mux_get_global_id(i32)

define spir_kernel void @insert_element(<4 x float> addrspace(1)* nocapture readonly %in, float %val, i32 %idx, <4 x float> addrspace(1)* nocapture %out) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0) #6
  %arrayidx = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %in, i64 %call
  %0 = bitcast <4 x float> addrspace(1)* %arrayidx to <4 x float> addrspace(1)*
  %1 = load <4 x float>, <4 x float> addrspace(1)* %0, align 16
  %2 = insertelement <4 x float> %1, float %val, i32 %idx
  %arrayidx3 = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %out, i64 %call
  store <4 x float> %2, <4 x float> addrspace(1)* %arrayidx3, align 4
  ret void
}

define spir_kernel void @insert_element_uniform(<4 x float> %in, float %val, i32 %idx, <4 x float> addrspace(1)* nocapture %out) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0) #6
  %0 = insertelement <4 x float> %in, float %val, i32 %idx
  %arrayidx3 = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %out, i64 %call
  store <4 x float> %0, <4 x float> addrspace(1)* %arrayidx3, align 4
  ret void
}

define spir_kernel void @insert_element_varying_indices(<4 x float> addrspace(1)* nocapture readonly %in, i32 addrspace(1)* %idxs, <4 x float> addrspace(1)* nocapture %out) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0) #6
  %arrayidxidx = getelementptr inbounds i32, i32 addrspace(1)* %idxs, i64 %call
  %idx = load i32, i32 addrspace(1)* %arrayidxidx
  %i = urem i32 %idx, 4
  %arrayidx = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %in, i64 %call
  %0 = bitcast <4 x float> addrspace(1)* %arrayidx to <4 x float> addrspace(1)*
  %1 = load <4 x float>, <4 x float> addrspace(1)* %0, align 16
  %fidx = uitofp i64 %call to float
  %2 = insertelement <4 x float> %1, float %fidx, i32 %i
  %arrayidx3 = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %out, i64 %call
  store <4 x float> %2, <4 x float> addrspace(1)* %arrayidx3, align 4
  ret void
}

define spir_kernel void @insert_element_bool(<4 x i32> addrspace(1)* %a, <4 x i32> addrspace(1)* %b, i32 %val, i32 %idx, <4 x i32> addrspace(1)* nocapture %out) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0) #6
  %arrayidxa = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %a, i64 %call
  %arrayidxb = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %b, i64 %call
  %0 = load <4 x i32>, <4 x i32> addrspace(1)* %arrayidxa, align 4
  %1 = load <4 x i32>, <4 x i32> addrspace(1)* %arrayidxb, align 4
  %2 = icmp slt <4 x i32> %0, %1
  %i = urem i64 %call, 4
  %v = trunc i32 %val to i1
  %3 = insertelement <4 x i1> %2, i1 %v, i64 %i
  %4 = sext <4 x i1> %3 to <4 x i32>
  %arrayidx4 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %out, i64 %call
  store <4 x i32> %4, <4 x i32> addrspace(1)* %arrayidx4, align 4
  ret void
}

; IE-LABEL: @__vecz_nxv4_insert_element(
; IE: [[ALLOC:%.*]] = alloca <vscale x 16 x float>, align 64
; IE: [[VAL0:%.*]] = insertelement <vscale x 4 x float> poison, float %val, {{(i32|i64)}} 0
; IE: [[VAL1:%.*]] = shufflevector <vscale x 4 x float> [[VAL0]], <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer
; IE: store <vscale x 16 x float> {{.*}}, ptr [[ALLOC]], align 64
; IE: [[IDX:%.*]] = sext i32 %idx to i64
; IE: [[ADDR:%.*]] = getelementptr inbounds float, ptr [[ALLOC]], i64 [[IDX]]
; IE: call void @__vecz_b_interleaved_store4_4_u5nxv4fu3ptr(<vscale x 4 x float> [[VAL1]], ptr nonnull [[ADDR]])
; IE: = load <vscale x 16 x float>, ptr [[ALLOC]], align 64

; Both the vector and index are uniform, so check we're not unnecessarily packetizing

; IE-UNI-LABEL: @__vecz_nxv4_insert_element_uniform(
; IE-UNI: {{%.*}} = insertelement <4 x float> %in, float %val, {{(i32|i64)}} %idx

; IE-INDICES-LABEL: @__vecz_nxv4_insert_element_varying_indices(
; IE-INDICES: [[ALLOC:%.*]] = alloca <vscale x 16 x float>, align 64
; IE-INDICES: [[VAL:%.*]] = uitofp <vscale x 4 x i64> {{%.*}} to <vscale x 4 x float>
; IE-INDICES: store <vscale x 16 x float> {{%.*}}, ptr [[ALLOC]], align 64
; IE-INDICES: [[T1:%.*]] = call <vscale x 4 x i32> @llvm.experimental.stepvector.nxv4i32()
; IE-INDICES: [[T2:%.*]] = shl <vscale x 4 x i32> [[T1]], shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> {{(undef|poison)}}, i32 2, {{(i32|i64)}} 0), <vscale x 4 x i32> {{(undef|poison)}}, <vscale x 4 x i32> zeroinitializer)

; LLVM 16 deduces add/or equivalence and uses `or` instead.
; IE-INDICES: [[T3:%.*]] = {{add|or}} {{(disjoint )?}}<vscale x 4 x i32> [[T2]], {{%.*}}

; IE-INDICES: [[T4:%.*]] = sext <vscale x 4 x i32> [[T3]] to <vscale x 4 x i64>
; IE-INDICES: [[ADDR:%.*]] = getelementptr inbounds float, ptr %0, <vscale x 4 x i64> [[T4]]
; IE-INDICES: call void @__vecz_b_scatter_store4_u5nxv4fu9nxv4u3ptr(<vscale x 4 x float> [[VAL]], <vscale x 4 x ptr> [[ADDR]])
; IE-INDICES: = load <vscale x 16 x float>, ptr [[ALLOC]], align 64

; Check we promote from i1 to i8 before doing our memops
; IE-BOOL-LABEL: @__vecz_nxv4_insert_element_bool(
; IE-BOOL: [[ALLOC:%.*]] = alloca <vscale x 16 x i8>, align 16
; IE-BOOL-DAG: [[T0:%.*]] = sext <vscale x 4 x i1> {{%.*}} to <vscale x 4 x i8>
; IE-BOOL-DAG: [[T1:%.*]] = sext <vscale x 16 x i1> {{%.*}} to <vscale x 16 x i8>
; IE-BOOL: store <vscale x 16 x i8> [[T1]], ptr [[ALLOC]], align 16
; IE-BOOL: call void @__vecz_b_scatter_store1_u5nxv4hu9nxv4u3ptr(<vscale x 4 x i8> [[T0]], <vscale x 4 x ptr> {{%.*}})
; IE-BOOL: [[T2:%.*]] = load <vscale x 16 x i8>, ptr [[ALLOC]], align 16
; IE-BOOL: [[T3:%.*]] = trunc <vscale x 16 x i8> [[T2]] to <vscale x 16 x i1>

