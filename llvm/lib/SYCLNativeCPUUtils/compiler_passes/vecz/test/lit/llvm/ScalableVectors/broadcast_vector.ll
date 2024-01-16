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

; RUN: veczc -vecz-scalable -vecz-simd-width=4 -vecz-passes="function(instcombine),packetizer,gvn,function(instcombine)" -S < %s | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare i64 @__mux_get_global_id(i32)

define dso_local spir_kernel void @vector_broadcast_const(<4 x float> addrspace(1)* nocapture readonly %in, <4 x float> addrspace(1)* nocapture %out) local_unnamed_addr #0 {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0) #6
  %arrayidx = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %in, i64 %call
  %0 = bitcast <4 x float> addrspace(1)* %arrayidx to <4 x float> addrspace(1)*
  %1 = load <4 x float>, <4 x float> addrspace(1)* %0, align 16
  %2 = fadd <4 x float> %1, <float 0x7FF8000020000000, float 0x7FF8000020000000, float 0x7FF8000020000000, float 0x7FF8000020000000>
  %arrayidx3 = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %out, i64 %call
  store <4 x float> %2, <4 x float> addrspace(1)* %arrayidx3, align 16
  ret void
}

define dso_local spir_kernel void @vector_broadcast(<4 x float> addrspace(1)* nocapture readonly %in, <4 x float> %addend, <4 x float> addrspace(1)* nocapture %out) local_unnamed_addr #0 {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0) #6
  %arrayidx = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %in, i64 %call
  %0 = bitcast <4 x float> addrspace(1)* %arrayidx to <4 x float> addrspace(1)*
  %1 = load <4 x float>, <4 x float> addrspace(1)* %0, align 16
  %2 = fadd <4 x float> %1, %addend
  %arrayidx3 = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %out, i64 %call
  store <4 x float> %2, <4 x float> addrspace(1)* %arrayidx3, align 16
  ret void
}

define dso_local spir_kernel void @vector_broadcast_regression(<4 x float> addrspace(1)* nocapture readonly %in, i32 %nancode, <4 x float> addrspace(1)* nocapture %out) local_unnamed_addr #0 {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0) #6
  %arrayidx = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %in, i64 %call
  %0 = bitcast <4 x float> addrspace(1)* %arrayidx to <4 x i32> addrspace(1)*
  %1 = load <4 x i32>, <4 x i32> addrspace(1)* %0, align 16
  %and1.i.i.i1.i = and <4 x i32> %1, <i32 2139095040, i32 2139095040, i32 2139095040, i32 2139095040>
  %cmp.i.i.i2.i = icmp ne <4 x i32> %and1.i.i.i1.i, <i32 2139095040, i32 2139095040, i32 2139095040, i32 2139095040>
  %and2.i.i.i3.i = and <4 x i32> %1, <i32 8388607, i32 8388607, i32 8388607, i32 8388607>
  %cmp3.i.i.i4.i = icmp eq <4 x i32> %and2.i.i.i3.i, zeroinitializer
  %2 = or <4 x i1> %cmp.i.i.i2.i, %cmp3.i.i.i4.i
  %3 = bitcast <4 x i32> %1 to <4 x float>
  %4 = select <4 x i1> %2, <4 x float> %3, <4 x float> <float 0x7FF0000020000000, float 0x7FF0000020000000, float 0x7FF0000020000000, float 0x7FF0000020000000>
  %arrayidx3 = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %out, i64 %call
  store <4 x float> %4, <4 x float> addrspace(1)* %arrayidx3, align 16
  ret void
}

; Check that new instructions aren't inserting before pre-existing allocas
define dso_local spir_kernel void @vector_broadcast_insertpt(<4 x float> addrspace(1)* nocapture readonly %in, <4 x float> %addend, i32 %nancode, <4 x float> addrspace(1)* nocapture %out, <4 x i32> addrspace(1)* nocapture %out2) local_unnamed_addr #0 {
entry:
  %existing.alloc = alloca <4 x i32>
  %call = tail call i64 @__mux_get_global_id(i32 0) #6
  store <4 x i32> zeroinitializer, <4 x i32>* %existing.alloc
  %scalar = bitcast <4 x i32>* %existing.alloc to i32*
  store i32 1, i32* %scalar
  %v = load <4 x i32>, <4 x i32>* %existing.alloc
  %arrayidx4 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %out2, i64 %call
  store <4 x i32> %v, <4 x i32> addrspace(1)* %arrayidx4, align 16

  %arrayidx = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %in, i64 %call
  %op = load <4 x float>, <4 x float> addrspace(1)* %arrayidx, align 16
  %v4 = fadd <4 x float> %op, %addend
  %arrayidx3 = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %out, i64 %call
  store <4 x float> %v4, <4 x float> addrspace(1)* %arrayidx3, align 16
  ret void
}

define dso_local spir_kernel void @vector_mask_broadcast(<4 x float> addrspace(1)* nocapture readonly %in, <4 x i1> %input, <4 x float> %woof, <4 x float> addrspace(1)* nocapture %out) local_unnamed_addr #0 {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0) #6
  %arrayidx = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %in, i64 %call
  %0 = bitcast <4 x float> addrspace(1)* %arrayidx to <4 x float> addrspace(1)*
  %1 = load <4 x float>, <4 x float> addrspace(1)* %0, align 16
  %2 = fcmp oeq <4 x float> %1, <float 1.0, float 1.0, float 1.0, float 1.0>
  %3 = and <4 x i1> %2, %input
  %4 = select <4 x i1> %3, <4 x float> %1, <4 x float> %woof
  %arrayidx3 = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %out, i64 %call
  store <4 x float> %4, <4 x float> addrspace(1)* %arrayidx3, align 16
  ret void
}
; CHECK-LABEL: @__vecz_nxv4_vector_broadcast_const(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CALL:%.*]] = tail call i64 @__mux_get_global_id(i32 0)
; CHECK-NEXT:    [[ARRAYIDX3:%.*]] = getelementptr inbounds <4 x float>, ptr addrspace(1) [[OUT:%.*]], i64 [[CALL]]
; CHECK-NEXT:    store <vscale x 16 x float> shufflevector (<vscale x 16 x float> insertelement (<vscale x 16 x float> {{(undef|poison)}}, float 0x7FF8000020000000, {{(i32|i64)}} 0), <vscale x 16 x float> {{(undef|poison)}}, <vscale x 16 x i32> zeroinitializer), ptr addrspace(1) [[ARRAYIDX3]], align 16
; CHECK-NEXT:    ret void

; CHECK-LABEL: @__vecz_nxv4_vector_broadcast(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[FIXLEN_ALLOC:%.*]] = alloca <4 x float>, align 16
; CHECK-NEXT:    store <4 x float> [[ADDEND:%.*]], ptr [[FIXLEN_ALLOC]], align 16
; CHECK-NEXT:    [[IDX0:%.*]] = call <vscale x 16 x i32> @llvm.experimental.stepvector.nxv16i32()
; CHECK-NEXT:    [[IDX1:%.*]] = and <vscale x 16 x i32> [[IDX0]], shufflevector (<vscale x 16 x i32> insertelement (<vscale x 16 x i32> {{(undef|poison)}}, i32 3, {{(i32|i64)}} 0), <vscale x 16 x i32> {{(undef|poison)}}, <vscale x 16 x i32> zeroinitializer)
; CHECK-NEXT:    [[TMP0:%.*]] = {{s|z}}ext{{( nneg)?}} <vscale x 16 x i32> [[IDX1]] to <vscale x 16 x i64>
; CHECK-NEXT:    [[VEC_ALLOC:%.*]] = getelementptr inbounds float, ptr [[FIXLEN_ALLOC]], <vscale x 16 x i64> [[TMP0]]
; CHECK-NEXT:    [[TMP1:%.*]] = call <vscale x 16 x float> @llvm.masked.gather.nxv16f32.nxv16p0(<vscale x 16 x ptr> [[VEC_ALLOC]], i32 4, <vscale x 16 x i1> shufflevector (<vscale x 16 x i1> insertelement (<vscale x 16 x i1> poison, i1 true, {{(i32|i64)}} 0), <vscale x 16 x i1> poison, <vscale x 16 x i32> zeroinitializer), <vscale x 16 x float> undef)
; CHECK-NEXT:    [[CALL:%.*]] = tail call i64 @__mux_get_global_id(i32 0)
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds <4 x float>, ptr addrspace(1) [[IN:%.*]], i64 [[CALL]]
; CHECK-NEXT:    [[TMP3:%.*]] = load <vscale x 16 x float>, ptr addrspace(1) [[ARRAYIDX]], align 16
; CHECK-NEXT:    [[TMP4:%.*]] = fadd <vscale x 16 x float> [[TMP3]], [[TMP1]]
; CHECK-NEXT:    [[ARRAYIDX3:%.*]] = getelementptr inbounds <4 x float>, ptr addrspace(1) [[OUT:%.*]], i64 [[CALL]]
; CHECK-NEXT:    store <vscale x 16 x float> [[TMP4]], ptr addrspace(1) [[ARRAYIDX3]], align 16
; CHECK-NEXT:    ret void

; CHECK-LABEL: @__vecz_nxv4_vector_broadcast_regression(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CALL:%.*]] = tail call i64 @__mux_get_global_id(i32 0)
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds <4 x float>, ptr addrspace(1) [[IN:%.*]], i64 [[CALL]]
; CHECK-NEXT:    [[TMP1:%.*]] = load <vscale x 16 x i32>, ptr addrspace(1) [[ARRAYIDX]], align 16
; CHECK-NEXT:    [[AND1_I_I_I1_I1:%.*]] = and <vscale x 16 x i32> [[TMP1]], shufflevector (<vscale x 16 x i32> insertelement (<vscale x 16 x i32> {{(undef|poison)}}, i32 2139095040, {{i32|i64}} 0), <vscale x 16 x i32> {{(undef|poison)}}, <vscale x 16 x i32> zeroinitializer)
; CHECK-NEXT:    [[CMP_I_I_I2_I2:%.*]] = icmp ne <vscale x 16 x i32> [[AND1_I_I_I1_I1]], shufflevector (<vscale x 16 x i32> insertelement (<vscale x 16 x i32> {{(undef|poison)}}, i32 2139095040, {{i32|i64}} 0), <vscale x 16 x i32> {{(undef|poison)}}, <vscale x 16 x i32> zeroinitializer)
; CHECK-NEXT:    [[AND2_I_I_I3_I3:%.*]] = and <vscale x 16 x i32> [[TMP1]], shufflevector (<vscale x 16 x i32> insertelement (<vscale x 16 x i32> {{(undef|poison)}}, i32 8388607, {{i32|i64}} 0), <vscale x 16 x i32> {{(undef|poison)}}, <vscale x 16 x i32> zeroinitializer)
; CHECK-NEXT:    [[CMP3_I_I_I4_I4:%.*]] = icmp eq <vscale x 16 x i32> [[AND2_I_I_I3_I3]], zeroinitializer
; CHECK-NEXT:    [[TMP2:%.*]] = or <vscale x 16 x i1> [[CMP_I_I_I2_I2]], [[CMP3_I_I_I4_I4]]
; CHECK-NEXT:    [[TMP3:%.*]] = bitcast <vscale x 16 x i32> [[TMP1]] to <vscale x 16 x float>
; CHECK-NEXT:    [[TMP4:%.*]] = select <vscale x 16 x i1> [[TMP2]], <vscale x 16 x float> [[TMP3]], <vscale x 16 x float> shufflevector (<vscale x 16 x float> insertelement (<vscale x 16 x float> {{(undef|poison)}}, float 0x7FF0000020000000, {{i32|i64}} 0), <vscale x 16 x float> {{(undef|poison)}}, <vscale x 16 x i32> zeroinitializer)
; CHECK-NEXT:    [[ARRAYIDX3:%.*]] = getelementptr inbounds <4 x float>, ptr addrspace(1) [[OUT:%.*]], i64 [[CALL]]
; CHECK-NEXT:    store <vscale x 16 x float> [[TMP4]], ptr addrspace(1) [[ARRAYIDX3]], align 16
; CHECK-NEXT:    ret void
;
;
; CHECK-LABEL: @__vecz_nxv4_vector_broadcast_insertpt(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[EXISTING_ALLOC:%.*]] = alloca <4 x i32>, align 16
; CHECK-NEXT:    [[FIXLEN_ALLOC:%.*]] = alloca <4 x i32>, align 16
; CHECK-NEXT:    [[FIXLEN_ALLOC1:%.*]] = alloca <4 x float>, align 16
; CHECK-NEXT:    store <4 x float> [[ADDEND:%.*]], ptr [[FIXLEN_ALLOC1]], align 16
; CHECK-NEXT:    [[IDX03:%.*]] = call <vscale x 16 x i32> @llvm.experimental.stepvector.nxv16i32()
; CHECK-NEXT:    [[IDX14:%.*]] = and <vscale x 16 x i32> [[IDX03]], shufflevector (<vscale x 16 x i32> insertelement (<vscale x 16 x i32> {{(undef|poison)}}, i32 3, {{i32|i64}} 0), <vscale x 16 x i32> {{(undef|poison)}}, <vscale x 16 x i32> zeroinitializer)
; CHECK-NEXT:    [[TMP0:%.*]] = {{s|z}}ext{{( nneg)?}} <vscale x 16 x i32> [[IDX14]] to <vscale x 16 x i64>
; CHECK-NEXT:    [[VEC_ALLOC5:%.*]] = getelementptr inbounds float, ptr [[FIXLEN_ALLOC1]], <vscale x 16 x i64> [[TMP0]]
; CHECK-NEXT:    [[TMP1:%.*]] = call <vscale x 16 x float> @llvm.masked.gather.nxv16f32.nxv16p0(<vscale x 16 x ptr> [[VEC_ALLOC5]], i32 4, <vscale x 16 x i1> shufflevector (<vscale x 16 x i1> insertelement (<vscale x 16 x i1> poison, i1 true, {{i32|i64}} 0), <vscale x 16 x i1> poison, <vscale x 16 x i32> zeroinitializer), <vscale x 16 x float> {{(undef|poison)}})
; CHECK-NEXT:    [[CALL:%.*]] = tail call i64 @__mux_get_global_id(i32 0)
; CHECK-NEXT:    store <4 x i32> zeroinitializer, ptr [[EXISTING_ALLOC]], align 16
; CHECK-NEXT:    store i32 1, ptr [[EXISTING_ALLOC]], align
; CHECK-NEXT:    [[V:%.*]] = load <4 x i32>, ptr [[EXISTING_ALLOC]], align 16
; CHECK-NEXT:    store <4 x i32> [[V]], ptr [[FIXLEN_ALLOC]], align 16
; CHECK-NEXT:    [[TMP2:%.*]] = {{s|z}}ext{{( nneg)?}} <vscale x 16 x i32> [[IDX14]] to <vscale x 16 x i64>
; CHECK-NEXT:    [[VEC_ALLOC:%.*]] = getelementptr inbounds i32, ptr [[FIXLEN_ALLOC]], <vscale x 16 x i64> [[TMP2]]
; CHECK-NEXT:    [[TMP3:%.*]] = call <vscale x 16 x i32> @llvm.masked.gather.nxv16i32.nxv16p0(<vscale x 16 x ptr> [[VEC_ALLOC]], i32 4, <vscale x 16 x i1> shufflevector (<vscale x 16 x i1> insertelement (<vscale x 16 x i1> poison, i1 true, {{i32|i64}} 0), <vscale x 16 x i1> poison, <vscale x 16 x i32> zeroinitializer), <vscale x 16 x i32> {{(undef|poison)}})
; CHECK-NEXT:    [[ARRAYIDX4:%.*]] = getelementptr inbounds <4 x i32>, ptr addrspace(1) [[OUT2:%.*]], i64 [[CALL]]
; CHECK-NEXT:    store <vscale x 16 x i32> [[TMP3]], ptr addrspace(1) [[ARRAYIDX4]], align 16
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds <4 x float>, ptr addrspace(1) [[IN:%.*]], i64 [[CALL]]
; CHECK-NEXT:    [[TMP6:%.*]] = load <vscale x 16 x float>, ptr addrspace(1) [[ARRAYIDX]], align 16
; CHECK-NEXT:    [[V46:%.*]] = fadd <vscale x 16 x float> [[TMP6]], [[TMP1]]
; CHECK-NEXT:    [[ARRAYIDX3:%.*]] = getelementptr inbounds <4 x float>, ptr addrspace(1) [[OUT:%.*]], i64 [[CALL]]
; CHECK-NEXT:    store <vscale x 16 x float> [[V46]], ptr addrspace(1) [[ARRAYIDX3]], align 16
; CHECK-NEXT:    ret void
;
; CHECK-LABEL: @__vecz_nxv4_vector_mask_broadcast(
; CHECK-NEXT:  entry:
; CHECK:    [[FIXLEN_MASK_ALLOC:%.*]] = alloca <4 x i8>, align 4
; CHECK:    [[IDX0:%.*]] = call <vscale x 16 x i32> @llvm.experimental.stepvector.nxv16i32()
; CHECK:    [[IDX1:%.*]] = and <vscale x 16 x i32> [[IDX0]], shufflevector (<vscale x 16 x i32> insertelement (<vscale x 16 x i32> {{(undef|poison)}}, i32 3, {{i32|i64}} 0), <vscale x 16 x i32> {{(undef|poison)}}, <vscale x 16 x i32> zeroinitializer)
; CHECK:    [[SEXT:%.*]] = sext <4 x i1> [[INPUT:%.*]] to <4 x i8>
; CHECK:    store <4 x i8> [[SEXT]], ptr [[FIXLEN_MASK_ALLOC]], align 4
; CHECK:    [[TMP0:%.*]] = {{s|z}}ext{{( nneg)?}} <vscale x 16 x i32> [[IDX1]] to <vscale x 16 x i64>
; CHECK:    [[VEC_ALLOC:%.*]] = getelementptr inbounds i8, ptr [[FIXLEN_MASK_ALLOC]], <vscale x 16 x i64> [[TMP0]]
; CHECK:    [[TMP1:%.*]] = call <vscale x 16 x i8> @llvm.masked.gather.nxv16i8.nxv16p0(<vscale x 16 x ptr> [[VEC_ALLOC]], i32 1, <vscale x 16 x i1> shufflevector (<vscale x 16 x i1> insertelement (<vscale x 16 x i1> poison, i1 true, {{i32|i64}} 0), <vscale x 16 x i1> poison, <vscale x 16 x i32> zeroinitializer), <vscale x 16 x i8> {{(undef|poison)}})
; CHECK:    [[BMASK:%.*]] = trunc <vscale x 16 x i8> [[TMP1]] to <vscale x 16 x i1>
; CHECK:    {{.*}} = and <vscale x 16 x i1> {{.*}}, [[BMASK]]
