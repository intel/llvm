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

; RUN: veczc -k extract_element -vecz-target-triple="riscv64-unknown-unknown" -vecz-scalable -vecz-simd-width=4 -S < %s | FileCheck %s --check-prefix=EE
; RUN: not veczc -k extract_element_ilegal -vecz-target-triple="riscv64-unknown-unknown" -vecz-scalable -vecz-simd-width=4 -S < %s
; RUN: veczc -k extract_element_uniform -vecz-target-triple="riscv64-unknown-unknown" -vecz-scalable -vecz-simd-width=4 -S < %s | FileCheck %s --check-prefix=EE-UNI
; RUN: veczc -k extract_element_uniform_vec -vecz-target-triple="riscv64-unknown-unknown" -vecz-scalable -vecz-simd-width=4 -S < %s | FileCheck %s --check-prefix=EE-UNI-VEC
; RUN: veczc -k extract_element_varying_indices -vecz-target-triple="riscv64-unknown-unknown" -vecz-scalable -vecz-simd-width=4 -S < %s | FileCheck %s --check-prefix=EE-INDICES
; RUN: veczc -k extract_element_bool -vecz-target-triple="riscv64-unknown-unknown" -vecz-scalable -vecz-simd-width=4 -S < %s | FileCheck %s --check-prefix=EE-BOOL

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

; NOTE: Base packetization failing for this case.

define spir_kernel void @extract_element_ilegal(<32 x float> addrspace(1)* nocapture readonly %in, i32 %idx, float addrspace(1)* nocapture %out) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0) #6
  %arrayidx = getelementptr inbounds <32 x float>, <32 x float> addrspace(1)* %in, i64 %call
  %0 = bitcast <32 x float> addrspace(1)* %arrayidx to <32 x float> addrspace(1)*
  %1 = load <32 x float>, <32 x float> addrspace(1)* %0, align 64
  %2 = extractelement <32 x float> %1, i32 %idx
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
; EE:         [[XLEN:%.*]] = call i64 @llvm.vscale.i64()
; EE-NEXT:    [[TMP2:%.*]] = shl i64 [[XLEN]], 2
; EE-NEXT:    [[SPLATINSERT:%.*]] = insertelement <vscale x 4 x i32> poison, i32 [[IDX:%.*]], {{(i32|i64)}} 0
; EE-NEXT:    [[SPLAT:%.*]] = shufflevector <vscale x 4 x i32> [[SPLATINSERT]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; EE-NEXT:    [[IDX0:%.*]] = call <vscale x 4 x i32> @llvm.experimental.stepvector.nxv4i32()
; EE-NEXT:    [[IDXSCALE:%.*]] = shl <vscale x 4 x i32> [[IDX0]], shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> poison, i32 2, {{(i32|i64)}} 0), <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer)
; EE-NEXT:    [[VS1:%.*]] = add <vscale x 4 x i32> [[IDXSCALE]], [[SPLAT]]
; EE-NEXT:    [[T3:%.*]] = call <vscale x 16 x i32> @llvm.{{(experimental.)?}}vector.insert.nxv16i32.nxv4i32(<vscale x 16 x i32> poison, <vscale x 4 x i32> [[VS1]], i64 0)
; EE-NEXT:    [[T4:%.*]] = call <vscale x 16 x float> @llvm.riscv.vrgather.vv.nxv16f32.i64(<vscale x 16 x float> poison, <vscale x 16 x float> [[T1:%.*]], <vscale x 16 x i32> [[T3]], i64 [[TMP2]])
; EE-NEXT:    [[T5:%.*]] = call <vscale x 4 x float> @llvm.{{(experimental.)?}}vector.extract.nxv4f32.nxv16f32(<vscale x 16 x float> [[T4]], i64 0)

; Both the vector and index are uniform, so check we're not unnecessarily packetizing 

; EE-UNI-LABEL: @__vecz_nxv4_extract_element_uniform(
; EE-UNI: [[T0:%.*]] = extractelement <4 x float> %in, i32 %idx
; EE-UNI: [[T1:%.*]] = insertelement <vscale x 4 x float> poison, float [[T0]], {{(i32|i64)}} 0
; EE-UNI: [[T2:%.*]] = shufflevector <vscale x 4 x float> [[T1]], <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer
; EE-UNI: store <vscale x 4 x float> [[T2]], ptr addrspace(1) {{%.*}}, align 4

; The vector is uniform and the index is varying, so we must broadcast the vector
; FIXME: Do we really need to broadcast? Can we mod the indices with the original vector length?

; EE-UNI-VEC-LABEL: @__vecz_nxv4_extract_element_uniform_vec(
; EE-UNI-VEC:         [[XLEN:%.*]] = call i64 @llvm.vscale.i64()
; EE-UNI-VEC:         [[T3:%.*]] = shl i64 [[XLEN]], 2
; EE-UNI-VEC-NEXT:    [[T:%.*]] = trunc <vscale x 4 x i64> [[T2:%.*]] to <vscale x 4 x i32>
; EE-UNI-VEC-NEXT:    [[I1:%.*]] = and <vscale x 4 x i32> [[T]], trunc (<vscale x 4 x i64> shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 3, {{(i32|i64)}} 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer) to <vscale x 4 x i32>)
; EE-UNI-VEC-NEXT:    [[IDX02:%.*]] = call <vscale x 4 x i32> @llvm.experimental.stepvector.nxv4i32()
; EE-UNI-VEC-NEXT:    [[IDXSCALE:%.*]] = shl <vscale x 4 x i32> [[IDX02]], shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> poison, i32 2, {{(i32|i64)}} 0), <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer)

; LLVM 16 deduces add/or equivalence and uses `or` instead.
; EE-UNI-VEC-NEXT:    [[VS1:%.*]] = {{add|or}} <vscale x 4 x i32> [[IDXSCALE]], [[I1]]

; EE-UNI-VEC-NEXT:    [[T4:%.*]] = call <vscale x 16 x i32> @llvm.{{(experimental.)?}}vector.insert.nxv16i32.nxv4i32(<vscale x 16 x i32> poison, <vscale x 4 x i32> [[VS1]], i64 0)
; EE-UNI-VEC-NEXT:    [[T5:%.*]] = call <vscale x 16 x float> @llvm.riscv.vrgather.vv.nxv16f32.i64(<vscale x 16 x float> poison, <vscale x 16 x float> [[T1:%.*]], <vscale x 16 x i32> [[T4]], i64 [[T3]])
; EE-UNI-VEC-NEXT:    [[T6:%.*]] = call <vscale x 4 x float> @llvm.{{(experimental.)?}}vector.extract.nxv4f32.nxv16f32(<vscale x 16 x float> [[T5]], i64 0)

; EE-INDICES-LABEL: @__vecz_nxv4_extract_element_varying_indices(
; EE-INDICES:         [[XLEN:%.*]] = call i64 @llvm.vscale.i64()
; EE-INDICES-NEXT:    [[T4:%.*]] = shl i64 [[XLEN]], 2
; EE-INDICES-NEXT:    [[IDX0:%.*]] = call <vscale x 4 x i32> @llvm.experimental.stepvector.nxv4i32()
; EE-INDICES-NEXT:    [[IDXSCALE:%.*]] = shl <vscale x 4 x i32> [[IDX0]], shufflevector (<vscale x 4 x i32> insertelement (<vscale x 4 x i32> poison, i32 2, {{(i32|i64)}} 0), <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer)
; EE-INDICES-NEXT:    [[VS1:%.*]] = {{add|or}} <vscale x 4 x i32> [[IDXSCALE]], [[I1:%.*]]
; EE-INDICES-NEXT:    [[T5:%.*]] = call <vscale x 16 x i32> @llvm.{{(experimental.)?}}vector.insert.nxv16i32.nxv4i32(<vscale x 16 x i32> poison, <vscale x 4 x i32> [[VS1]], i64 0)
; EE-INDICES-NEXT:    [[T6:%.*]] = call <vscale x 16 x float> @llvm.riscv.vrgather.vv.nxv16f32.i64(<vscale x 16 x float> poison, <vscale x 16 x float> [[T3:%.*]], <vscale x 16 x i32> [[T5]], i64 [[T4]])
; EE-INDICES-NEXT:    [[T7:%.*]] = call <vscale x 4 x float> @llvm.{{(experimental.)?}}vector.extract.nxv4f32.nxv16f32(<vscale x 16 x float> [[T6]], i64 0)

; Check we promote from i1 to i8 before doing our memops and use vrgatherei16.
; EE-BOOL-LABEL: @__vecz_nxv4_extract_element_bool(
; EE-BOOL:       [[T6:%.*]] = sext <vscale x 16 x i1> [[T5:%.*]] to <vscale x 16 x i8>
; EE-BOOL-NEXT:  [[XLEN:%.*]] = call i64 @llvm.vscale.i64()
; EE-BOOL-NEXT:  [[T7:%.*]] = shl i64 [[XLEN]], 2
; EE-BOOL-NEXT:  [[T8:%.*]] = trunc <vscale x 4 x i64> [[T0:%.*]] to <vscale x 4 x i16>
; EE-BOOL-NEXT:  [[T9:%.*]] = and <vscale x 4 x i16> [[T8]], trunc (<vscale x 4 x i64> shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 3, {{(i32|i64)}} 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer) to <vscale x 4 x i16>)
; EE-BOOL-NEXT:  [[T10:%.*]] = call <vscale x 4 x i16> @llvm.experimental.stepvector.nxv4i16()
; EE-BOOL-NEXT:  [[T11:%.*]] = shl <vscale x 4 x i16> [[T10]], shufflevector (<vscale x 4 x i16> insertelement (<vscale x 4 x i16> poison, i16 2, {{(i32|i64)}} 0), <vscale x 4 x i16> poison, <vscale x 4 x i32> zeroinitializer)
; EE-BOOL-NEXT:  [[VS1:%.*]] = {{add|or}} <vscale x 4 x i16> [[T11]], [[T9]]
; EE-BOOL-NEXT:  [[T12:%.*]] = call <vscale x 16 x i16> @llvm.{{(experimental.)?}}vector.insert.nxv16i16.nxv4i16(<vscale x 16 x i16> poison, <vscale x 4 x i16> [[VS1]], i64 0)
; EE-BOOL-NEXT:  [[T13:%.*]] = call <vscale x 16 x i8> @llvm.riscv.vrgatherei16.vv.nxv16i8.i64(<vscale x 16 x i8> poison, <vscale x 16 x i8> [[T6]], <vscale x 16 x i16> [[T12]], i64 [[T7]])
; EE-BOOL-NEXT:  [[T14:%.*]] = call <vscale x 4 x i8> @llvm.{{(experimental.)?}}vector.extract.nxv4i8.nxv16i8(<vscale x 16 x i8> [[T13]], i64 0)
; EE-BOOL-NEXT:  [[T15:%.*]] = trunc <vscale x 4 x i8> [[T14]] to <vscale x 4 x i1>
