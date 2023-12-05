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
; RUN: veczc -k extract_element -vecz-scalable -vecz-simd-width=4 -S < %s | FileCheck %s --check-prefix=EE
; RUN: veczc -k extract_element_uniform -vecz-scalable -vecz-simd-width=4 -S < %s | FileCheck %s --check-prefix=EE-UNI
; RUN: veczc -k extract_element_uniform_vec -vecz-scalable -vecz-simd-width=4 -S < %s | FileCheck %s --check-prefix=EE-UNI-VEC
; RUN: veczc -k extract_element_varying_indices -vecz-scalable -vecz-simd-width=4 -S < %s | FileCheck %s --check-prefix=EE-INDICES
; RUN: veczc -k extract_element_bool -vecz-scalable -vecz-simd-width=4 -S < %s | FileCheck %s --check-prefix=EE-BOOL

target triple = "spir64-unknown-unknown"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare i64 @__mux_get_global_id(i32)

define spir_kernel void @extract_element(<4 x float> addrspace(1)* nocapture readonly %in, i32 %idx, float addrspace(1)* nocapture %out) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0) #6
  %arrayidx = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %in, i64 %call
  %0 = bitcast <4 x float> addrspace(1)* %arrayidx to <4 x float> addrspace(1)*
  %1 = load <4 x float>, <4 x float> addrspace(1)* %0, align 16
  %2 = extractelement <4 x float> %1, i32 %idx
  %arrayidx3 = getelementptr inbounds float, float addrspace(1)* %out, i64 %call
  store float %2, float addrspace(1)* %arrayidx3, align 4
  ret void
}

define spir_kernel void @extract_element_uniform(<4 x float> %in, i32 %idx, float addrspace(1)* nocapture %out) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0) #6
  %0 = extractelement <4 x float> %in, i32 %idx
  %arrayidx3 = getelementptr inbounds float, float addrspace(1)* %out, i64 %call
  store float %0, float addrspace(1)* %arrayidx3, align 4
  ret void
}

define spir_kernel void @extract_element_uniform_vec(<4 x float> %in, float addrspace(1)* nocapture %out) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0) #6
  %i = urem i64 %call, 4
  %0 = extractelement <4 x float> %in, i64 %i
  %arrayidx3 = getelementptr inbounds float, float addrspace(1)* %out, i64 %call
  store float %0, float addrspace(1)* %arrayidx3, align 4
  ret void
}

define spir_kernel void @extract_element_varying_indices(<4 x float> addrspace(1)* %in, i32 addrspace(1)* %idxs, float addrspace(1)* nocapture %out) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0)
  %arrayidxidx = getelementptr inbounds i32, i32 addrspace(1)* %idxs, i64 %call
  %idx = load i32, i32 addrspace(1)* %arrayidxidx
  %i = urem i32 %idx, 4
  %arrayidx = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %in, i64 %call
  %0 = load <4 x float>, <4 x float> addrspace(1)* %arrayidx
  %1 = extractelement <4 x float> %0, i32 %i
  %arrayidx3 = getelementptr inbounds float, float addrspace(1)* %out, i64 %call
  store float %1, float addrspace(1)* %arrayidx3, align 4
  ret void
}

define spir_kernel void @extract_element_bool(<4 x i32> addrspace(1)* %a, <4 x i32> addrspace(1)* %b, i32 %idx, i32 addrspace(1)* nocapture %out, <4 x i32> addrspace(1)* nocapture %out2) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0) #6
  %arrayidxa = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %a, i64 %call
  %arrayidxb = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %b, i64 %call
  %0 = load <4 x i32>, <4 x i32> addrspace(1)* %arrayidxa, align 4
  %1 = load <4 x i32>, <4 x i32> addrspace(1)* %arrayidxb, align 4
  %2 = icmp slt <4 x i32> %0, %1
  %i = urem i64 %call, 4
  %3 = extractelement <4 x i1> %2, i64 %i
  %4 = sext i1 %3 to i32
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %4, i32 addrspace(1)* %arrayidx3, align 4
  %5 = sext <4 x i1> %2 to <4 x i32>
  %arrayidx4 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %out2, i64 %call
  store <4 x i32> %5, <4 x i32> addrspace(1)* %arrayidx4, align 4
  ret void
}


; EE-LABEL: @__vecz_nxv4_extract_element(
; EE: [[ALLOC:%.*]] = alloca <vscale x 16 x float>, align 64
; EE: store <vscale x 16 x float> {{.*}}, ptr [[ALLOC]], align 64
; EE: [[IDX:%.*]] = sext i32 %idx to i64
; EE: [[ADDR:%.*]] = getelementptr inbounds float, ptr [[ALLOC]], i64 [[IDX]]
; EE: [[GATHER:%.*]] = call <vscale x 4 x float> @__vecz_b_interleaved_load4_4_u5nxv4fu3ptr(ptr nonnull [[ADDR]])

; Both the vector and index are uniform, so check we're not unnecessarily packetizing 

; EE-UNI-LABEL: @__vecz_nxv4_extract_element_uniform(
; EE-UNI: [[T0:%.*]] = extractelement <4 x float> %in, i32 %idx
; EE-UNI: [[T1:%.*]] = insertelement <vscale x 4 x float> poison, float [[T0]], {{(i32|i64)}} 0
; EE-UNI: [[T2:%.*]] = shufflevector <vscale x 4 x float> [[T1]], <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer
; EE-UNI: store <vscale x 4 x float> [[T2]], ptr addrspace(1) {{%.*}}, align 4

; The vector is uniform and the index is varying, so we must broadcast the vector
; FIXME: Do we really need to broadcast? Can we mod the indices with the original vector length?

; EE-UNI-VEC-LABEL: @__vecz_nxv4_extract_element_uniform_vec(
; EE-UNI-VEC: [[T3:%.*]] = insertelement <vscale x 4 x i64> poison, i64 %call, {{(i32|i64)}} 0
; EE-UNI-VEC: [[T4:%.*]] = shufflevector <vscale x 4 x i64> [[T3]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; EE-UNI-VEC: [[STEP:%.*]] = call <vscale x 4 x i64> @llvm.experimental.stepvector.nxv4i64()
; EE-UNI-VEC: [[T5:%.*]] = add <vscale x 4 x i64> [[T4]], [[STEP]]
; EE-UNI-VEC: [[MOD:%.*]] = and <vscale x 4 x i64> [[T5]], shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> {{(undef|poison)}}, i64 3, {{(i32|i64)}} 0), <vscale x 4 x i64> {{(undef|poison)}}, <vscale x 4 x i32> zeroinitializer)
; EE-UNI-VEC: [[T6:%.*]] = shl <vscale x 4 x i64> [[STEP]], shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> {{(undef|poison)}}, i64 2, {{(i32|i64)}} 0), <vscale x 4 x i64> {{(undef|poison)}}, <vscale x 4 x i32> zeroinitializer)

; LLVM 16 deduces add/or equivalence and uses `or` instead.
; EE-UNI-VEC: [[T7:%.*]] = {{add|or}} {{(disjoint )?}}<vscale x 4 x i64> [[T6]], [[MOD]]

; EE-UNI-VEC: [[T8:%.*]] = getelementptr inbounds float, ptr {{%.*}}, <vscale x 4 x i64> [[T7]]
; EE-UNI-VEC: [[T9:%.*]] = call <vscale x 4 x float> @__vecz_b_gather_load4_u5nxv4fu9nxv4u3ptr(<vscale x 4 x ptr> [[T8]])
; EE-UNI-VEC: store <vscale x 4 x float> [[T9]], ptr addrspace(1) {{%.*}}, align 4

; EE-INDICES-LABEL: @__vecz_nxv4_extract_element_varying_indices(
; EE-INDICES: [[ALLOC:%.*]] = alloca <vscale x 16 x float>, align 64
; EE-INDICES: [[T0:%.*]] = getelementptr inbounds i32, ptr addrspace(1) %idxs, i64 %call
; EE-INDICES: [[T2:%.*]] = load <vscale x 4 x i32>, ptr addrspace(1) [[T0]], align 4
; EE-INDICES: [[T3:%.*]] = and <vscale x 4 x i32> [[T2]], shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> {{(undef|poison)}}, i32 3, {{i32|i64}} 0), <vscale x 4 x i32> {{(undef|poison)}}, <vscale x 4 x i32> zeroinitializer)
; EE-INDICES: store <vscale x 16 x float> {{.*}}, ptr [[ALLOC]], align 64
; EE-INDICES: [[STEP:%.*]] = call <vscale x 4 x i32> @llvm.experimental.stepvector.nxv4i32()
; EE-INDICES: [[T4:%.*]] = shl <vscale x 4 x i32> [[STEP]], shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> {{(undef|poison)}}, i32 2, {{i32|i64}} 0), <vscale x 4 x i32> {{(undef|poison)}}, <vscale x 4 x i32> zeroinitializer)
; EE-INDICES: [[T5:%.*]] = {{add|or}} {{(disjoint )?}}<vscale x 4 x i32> [[T4]], [[T3]]
; EE-INDICES: [[IDX:%.*]] = sext <vscale x 4 x i32> [[T5]] to <vscale x 4 x i64>
; EE-INDICES: [[ADDR:%.*]] = getelementptr inbounds float, ptr [[ALLOC]], <vscale x 4 x i64> [[IDX]]
; EE-INDICES: [[GATHER:%.*]] = call <vscale x 4 x float> @__vecz_b_gather_load4_u5nxv4fu9nxv4u3ptr(<vscale x 4 x ptr> [[ADDR]])

; Check we promote from i1 to i8 before doing our memops
; EE-BOOL-LABEL: @__vecz_nxv4_extract_element_bool(
; EE-BOOL: [[T0:%.*]] = sext <vscale x 16 x i1> {{%.*}} to <vscale x 16 x i8>
; EE-BOOL: store <vscale x 16 x i8> {{.*}}
; EE-BOOL: [[T1:%.*]] = call <vscale x 4 x i8> @__vecz_b_gather_load1_u5nxv4hu9nxv4u3ptr(<vscale x 4 x ptr> {{%.*}}
; EE-BOOL: [[T2:%.*]] = trunc <vscale x 4 x i8> [[T1]] to <vscale x 4 x i1>
