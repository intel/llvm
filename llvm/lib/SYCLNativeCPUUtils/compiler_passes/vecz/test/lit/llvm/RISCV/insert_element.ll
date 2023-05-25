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
; RUN: veczc -k insert_element -vecz-target-triple="riscv64-unknown-unknown" -vecz-scalable -vecz-simd-width=4 -S < %s | FileCheck %s --check-prefix=IE
; RUN: veczc -k insert_element_uniform -vecz-target-triple="riscv64-unknown-unknown" -vecz-scalable -vecz-simd-width=4 -S < %s | FileCheck %s --check-prefix=IE-UNI
; RUN: veczc -k insert_element_varying_indices -vecz-target-triple="riscv64-unknown-unknown" -vecz-scalable -vecz-simd-width=4 -S < %s | FileCheck %s --check-prefix=IE-INDICES
; RUN: not veczc -k insert_element_illegal -vecz-target-triple="riscv64-unknown-unknown" -vecz-scalable -vecz-simd-width=4 -S < %s
; RUN: veczc -k insert_element_bool -vecz-target-triple="riscv64-unknown-unknown" -vecz-scalable -vecz-simd-width=4 -S < %s | FileCheck %s --check-prefix=IE-BOOL

target triple = "spir64-unknown-unknown"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare spir_func i64 @_Z13get_global_idj(i32)

define spir_kernel void @insert_element(<4 x float> addrspace(1)* nocapture readonly %in, float %val, i32 %idx, <4 x float> addrspace(1)* nocapture %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0) #6
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
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0) #6
  %0 = insertelement <4 x float> %in, float %val, i32 %idx
  %arrayidx3 = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %out, i64 %call
  store <4 x float> %0, <4 x float> addrspace(1)* %arrayidx3, align 4
  ret void
}

define spir_kernel void @insert_element_varying_indices(<4 x float> addrspace(1)* nocapture readonly %in, i32 addrspace(1)* %idxs, <4 x float> addrspace(1)* nocapture %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0) #6
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

define spir_kernel void @insert_element_illegal(<32 x float> addrspace(1)* nocapture readonly %in, i32 addrspace(1)* %idxs, <32 x float> addrspace(1)* nocapture %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0) #6
  %arrayidxidx = getelementptr inbounds i32, i32 addrspace(1)* %idxs, i64 %call
  %idx = load i32, i32 addrspace(1)* %arrayidxidx, align 4
  %i = urem i32 %idx, 32
  %arrayidx = getelementptr inbounds <32 x float>, <32 x float> addrspace(1)* %in, i64 %call
  %0 = bitcast <32 x float> addrspace(1)* %arrayidx to <32 x float> addrspace(1)*
  %1 = load <32 x float>, <32 x float> addrspace(1)* %0, align 64
  %fidx = uitofp i64 %call to float
  %2 = insertelement <32 x float> %1, float %fidx, i32 %i
  %arrayidx3 = getelementptr inbounds <32 x float>, <32 x float> addrspace(1)* %out, i64 %call
  store <32 x float> %2, <32 x float> addrspace(1)* %arrayidx3, align 64
  ret void
}

define spir_kernel void @insert_element_bool(<4 x i32> addrspace(1)* %a, <4 x i32> addrspace(1)* %b, i32 %val, i32 %idx, <4 x i32> addrspace(1)* nocapture %out) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0) #6
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
; IE:         [[SPLATINSERT:%.*]] = insertelement <vscale x 4 x float> poison, float [[VAL:%.*]], {{(i32|i64)}} 0
; IE:         [[SPLAT:%.*]] = shufflevector <vscale x 4 x float> [[SPLATINSERT]], <vscale x 4 x float> poison, <vscale x 4 x i32> zeroinitializer
; IE:         [[XLEN:%.*]] = call i64 @llvm.vscale.i64()
; IE-NEXT:    [[TMP2:%.*]] = shl i64 [[XLEN]], 4
; IE-NEXT:    [[SPLATINSERT1:%.*]] = insertelement <vscale x 16 x i32> poison, i32 [[IDX:%.*]], {{(i32|i64)}} 0
; IE-NEXT:    [[SPLAT2:%.*]] = shufflevector <vscale x 16 x i32> [[SPLATINSERT1]], <vscale x 16 x i32> poison, <vscale x 16 x i32> zeroinitializer
; IE-NEXT:    [[ELTS:%.*]] = call <vscale x 16 x float> @llvm.{{(experimental.)?}}vector.insert.nxv16f32.nxv4f32(<vscale x 16 x float> poison, <vscale x 4 x float> [[SPLAT]], i64 0)
; IE-NEXT:    [[STEP:%.*]] = call <vscale x 16 x i32> @llvm.experimental.stepvector.nxv16i32()
; IE-NEXT:    [[INNER:%.*]] = and <vscale x 16 x i32> [[STEP]], shufflevector (<vscale x 16 x i32> insertelement (<vscale x 16 x i32> poison, i32 3, {{i32|i64}} 0), <vscale x 16 x i32> poison, <vscale x 16 x i32> zeroinitializer)
; IE-NEXT:    [[OUTER:%.*]] = lshr <vscale x 16 x i32> [[STEP]], shufflevector (<vscale x 16 x i32> insertelement (<vscale x 16 x i32> poison, i32 2, {{i32|i64}} 0), <vscale x 16 x i32> poison, <vscale x 16 x i32> zeroinitializer)
; IE-NEXT:    [[VM:%.*]] = icmp eq <vscale x 16 x i32> [[SPLAT2]], [[INNER]]
; IE-NEXT:    [[TMP8:%.*]] = call <vscale x 16 x float> @llvm.riscv.vrgather.vv.mask.nxv16f32.i64(<vscale x 16 x float> [[TMP1:%.*]], <vscale x 16 x float> [[ELTS]], <vscale x 16 x i32> [[OUTER]], <vscale x 16 x i1> [[VM]], i64 [[TMP2]]{{(, i64 1)?}})

; Both the vector and index are uniform, so check we're not unnecessarily packetizing

; IE-UNI-LABEL: @__vecz_nxv4_insert_element_uniform(
; IE-UNI: {{%.*}} = insertelement <4 x float> %in, float %val, {{(i32|i64)}} %idx

; IE-INDICES-LABEL: @__vecz_nxv4_insert_element_varying_indices(
; IE-INDICES:         [[FIDX2:%.*]] = uitofp <vscale x 4 x i64> [[TMP0:%.*]] to <vscale x 4 x float>
; IE-INDICES-NEXT:    [[XLEN:%.*]] = call i64 @llvm.vscale.i64()
; IE-INDICES-NEXT:    [[TMP5:%.*]] = shl i64 [[XLEN]], 4
; IE-INDICES-NEXT:    [[VS2:%.*]] = call <vscale x 16 x i32> @llvm.{{(experimental.)?}}vector.insert.nxv16i32.nxv4i32(<vscale x 16 x i32> poison, <vscale x 4 x i32> {{%.*}}, i64 0)
; IE-INDICES:         [[IDX0:%.*]] = call <vscale x 16 x i32> @llvm.experimental.stepvector.nxv16i32()
; IE-INDICES-NEXT:    [[IDX1:%.*]] = lshr <vscale x 16 x i32> [[IDX0]], shufflevector (<vscale x 16 x i32> insertelement (<vscale x 16 x i32> {{(undef|poison)}}, i32 2, {{(i32|i64)}} 0), <vscale x 16 x i32> {{(undef|poison)}}, <vscale x 16 x i32> zeroinitializer)
; IE-INDICES-NEXT:    [[TMP9:%.*]] = call <vscale x 16 x i32> @llvm.riscv.vrgather.vv.nxv16i32.i64(<vscale x 16 x i32> undef, <vscale x 16 x i32> [[VS2:%.*]], <vscale x 16 x i32> [[IDX1]], i64 [[TMP5]])
; IE-INDICES-NEXT:    [[VS25:%.*]] = call <vscale x 16 x float> @llvm.{{(experimental.)?}}vector.insert.nxv16f32.nxv4f32(<vscale x 16 x float> poison, <vscale x 4 x float> [[FIDX2]], i64 0)
; IE-INDICES-NEXT:    [[INNER:%.*]] = and <vscale x 16 x i32> [[IDX0]], shufflevector (<vscale x 16 x i32> insertelement (<vscale x 16 x i32> poison, i32 3, {{i32|i64}} 0), <vscale x 16 x i32> poison, <vscale x 16 x i32> zeroinitializer)
; IE-INDICES-NEXT:    [[VM:%.*]] = icmp eq <vscale x 16 x i32> [[TMP9]], [[INNER]]
; IE-INDICES-NEXT:    [[TMP11:%.*]] = call <vscale x 16 x float> @llvm.riscv.vrgather.vv.mask.nxv16f32.i64(<vscale x 16 x float> [[TMP4:%.*]], <vscale x 16 x float> [[VS25]], <vscale x 16 x i32> [[IDX1]], <vscale x 16 x i1> [[VM]], i64 [[TMP5]]{{(, i64 1)?}})

; Check we promote from i1 to i8 before doing our memops
; IE-BOOL-LABEL: @__vecz_nxv4_insert_element_bool(
; IE-BOOL-DAG:     [[T1:%.*]] = sext <vscale x 16 x i1> {{%.*}} to <vscale x 16 x i8>
; IE-BOOL-DAG:     [[T0:%.*]] = sext <vscale x 4 x i1> {{%.*}} to <vscale x 4 x i8>
; IE-BOOL:         [[TMP18:%.*]] = call <vscale x 16 x i8> @llvm.riscv.vrgatherei16.vv.mask.nxv16i8.i64(<vscale x 16 x i8> [[TMP7:%.*]], <vscale x 16 x i8> {{%.*}}, <vscale x 16 x i16> [[TMP16:%.*]], <vscale x 16 x i1> [[VM:%.*]], i64 [[TMP8:%.*]])
;                            %12 = call <vscale x 16 x i8> @llvm.riscv.vrgatherei16.vv.mask.nxv16i8.i64(<vscale x 16 x i8> %6, <vscale x 16 x i8> %vs25, <vscale x 16 x i16> %vs16, <vscale x 16 x i1> %vm, i64 %7, i64 1)
; IE-BOOL-NEXT:    [[TMP19:%.*]] = trunc <vscale x 16 x i8> [[TMP18]] to <vscale x 16 x i1>
