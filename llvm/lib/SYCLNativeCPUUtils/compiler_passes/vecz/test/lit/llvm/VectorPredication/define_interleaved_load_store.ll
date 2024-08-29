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
; RUN: veczc -k f -vecz-scalable -vecz-simd-width=4 -vecz-choices=VectorPredication:FullScalarization -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @f(<4 x double> addrspace(1)* %a, <4 x double> addrspace(1)* %b, <4 x double> addrspace(1)* %c, <4 x double> addrspace(1)* %d, <4 x double> addrspace(1)* %e, i8 addrspace(1)* %flag) {
entry:
  %call = call i64 @__mux_get_global_id(i32 0)
  %add.ptr = getelementptr inbounds <4 x double>, <4 x double> addrspace(1)* %b, i64 %call
  %.cast = getelementptr inbounds <4 x double>, <4 x double> addrspace(1)* %add.ptr, i64 0, i64 0
  %0 = load <4 x double>, <4 x double> addrspace(1)* %add.ptr, align 32
  store double 1.600000e+01, double addrspace(1)* %.cast, align 8
  %1 = load <4 x double>, <4 x double> addrspace(1)* %add.ptr, align 32
  %vecins5 = shufflevector <4 x double> %0, <4 x double> %1, <4 x i32> <i32 0, i32 1, i32 6, i32 undef>
  %vecins7 = shufflevector <4 x double> %vecins5, <4 x double> %1, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  %arrayidx = getelementptr inbounds <4 x double>, <4 x double> addrspace(1)* %c, i64 %call
  %2 = load <4 x double>, <4 x double> addrspace(1)* %arrayidx, align 32
  %arrayidx8 = getelementptr inbounds <4 x double>, <4 x double> addrspace(1)* %d, i64 %call
  %3 = load <4 x double>, <4 x double> addrspace(1)* %arrayidx8, align 32
  %arrayidx9 = getelementptr inbounds <4 x double>, <4 x double> addrspace(1)* %e, i64 %call
  %4 = load <4 x double>, <4 x double> addrspace(1)* %arrayidx9, align 32
  %div = fdiv <4 x double> %3, %4
  %5 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %vecins7, <4 x double> %2, <4 x double> %div)
  %arrayidx10 = getelementptr inbounds <4 x double>, <4 x double> addrspace(1)* %a, i64 %call
  %6 = load <4 x double>, <4 x double> addrspace(1)* %arrayidx10, align 32
  %sub = fsub <4 x double> %6, %5
  store <4 x double> %sub, <4 x double> addrspace(1)* %arrayidx10, align 32
  ret void
}

declare i64 @__mux_get_global_id(i32) #1

declare void @__mux_work_group_barrier(i32, i32, i32) #1

; Function Attrs: nounwind readnone
declare <4 x double> @llvm.fmuladd.v4f64(<4 x double>, <4 x double>, <4 x double>) #2

; Test if the interleaved load is defined correctly
; Vector-predicated interleaved loads are always masked
; CHECK: define <vscale x 4 x double> @__vecz_b_masked_interleaved_load8_vp_4_u5nxv4du3ptrU3AS1u5nxv4bj(ptr addrspace(1){{( %0)?}}, <vscale x 4 x i1>{{( %1)?}}, i32{{( %2)?}}) [[ATTRS:#[0-9]+]] {
; CHECK: entry:
; CHECK:   %BroadcastAddr.splatinsert = insertelement <vscale x 4 x ptr addrspace(1)> poison, ptr addrspace(1) %0, {{i32|i64}} 0
; CHECK:   %BroadcastAddr.splat = shufflevector <vscale x 4 x ptr addrspace(1)> %BroadcastAddr.splatinsert, <vscale x 4 x ptr addrspace(1)> poison, <vscale x 4 x i32> zeroinitializer
; CHECK:   %3 = call <vscale x 4 x i64> @llvm.{{(experimental\.)?}}stepvector.nxv4i64()
; CHECK:   %4 = mul <vscale x 4 x i64> shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 4, {{i32|i64}} 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer), %3
; CHECK:   %5 = getelementptr double, <vscale x 4 x ptr addrspace(1)> %BroadcastAddr.splat, <vscale x 4 x i64> %4
; CHECK:   %6 = call <vscale x 4 x double> @llvm.vp.gather.nxv4f64.nxv4p1(<vscale x 4 x ptr addrspace(1)> %5, <vscale x 4 x i1> %1, i32 %2)
; CHECK:   ret <vscale x 4 x double> %6
; CHECK: }


; Test if the interleaved store is defined correctly
; Vector-predicated interleaved stores are always masked
; CHECK: define void @__vecz_b_masked_interleaved_store8_vp_4_u5nxv4du3ptrU3AS1u5nxv4bj(<vscale x 4 x double>{{( %0)?}}, ptr addrspace(1){{( %1)?}}, <vscale x 4 x i1>{{( %2)?}}, i32{{( %3)?}}) [[ATTRS]]
; CHECK: entry:
; CHECK:  %BroadcastAddr.splatinsert = insertelement <vscale x 4 x ptr addrspace(1)> poison, ptr addrspace(1) %1, {{i32|i64}} 0
; CHECK:  %BroadcastAddr.splat = shufflevector <vscale x 4 x ptr addrspace(1)> %BroadcastAddr.splatinsert, <vscale x 4 x ptr addrspace(1)> poison, <vscale x 4 x i32> zeroinitializer
; CHECK:  %4 = call <vscale x 4 x i64> @llvm.{{(experimental\.)?}}stepvector.nxv4i64()
; CHECK:  %5 = mul <vscale x 4 x i64> shufflevector (<vscale x 4 x i64> insertelement (<vscale x 4 x i64> poison, i64 4, {{i32|i64}} 0), <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer), %4
; CHECK:  %6 = getelementptr double, <vscale x 4 x ptr addrspace(1)> %BroadcastAddr.splat, <vscale x 4 x i64> %5
; CHECK:  call void @llvm.vp.scatter.nxv4f64.nxv4p1(<vscale x 4 x double> %0, <vscale x 4 x ptr addrspace(1)> %6, <vscale x 4 x i1> %2, i32 %3)
; CHECK:  ret void
; CHECK: }

; CHECK: attributes [[ATTRS]] = { norecurse nounwind }
