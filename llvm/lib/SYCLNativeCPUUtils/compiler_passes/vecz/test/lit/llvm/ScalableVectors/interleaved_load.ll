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
; RUN: veczc -k load_interleaved -vecz-scalable -vecz-simd-width=4 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_kernel void @load_interleaved(i32 addrspace(1)* nocapture readonly %input, i32 addrspace(1)* nocapture %output, i32 %stride) local_unnamed_addr {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0) #2
  %0 = trunc i64 %call to i32
  %conv1 = mul i32 %0, %stride
  %idxprom = sext i32 %conv1 to i64
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %input, i64 %idxprom
  %1 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %output, i64 %idxprom
  store i32 %1, i32 addrspace(1)* %arrayidx3, align 4
  %add = add nsw i32 %conv1, 1
  %idxprom4 = sext i32 %add to i64
  %arrayidx5 = getelementptr inbounds i32, i32 addrspace(1)* %output, i64 %idxprom4
  store i32 1, i32 addrspace(1)* %arrayidx5, align 4
  %add6 = add nsw i32 %conv1, 2
  %idxprom7 = sext i32 %add6 to i64
  %arrayidx8 = getelementptr inbounds i32, i32 addrspace(1)* %output, i64 %idxprom7
  store i32 1, i32 addrspace(1)* %arrayidx8, align 4
  ret void
}

declare i64 @__mux_get_global_id(i32)

; CHECK: define void @__vecz_b_interleaved_store4_V_u5nxv4ju3ptrU3AS1(<vscale x 4 x i32> [[ARG0:%.*]], ptr addrspace(1) [[ARG1:%.*]], i64 [[ARG2:%.*]]) [[ATTRS:#[0-9]+]] {
; CHECK-NEXT: entry:
; CHECK-NEXT: [[TMP0:%.*]] = insertelement <vscale x 4 x ptr addrspace(1)> poison, ptr addrspace(1) [[ARG1]], {{i32|i64}} 0
; CHECK-NEXT: [[TMP1:%.*]] = shufflevector <vscale x 4 x ptr addrspace(1)> [[TMP0]], <vscale x 4 x ptr addrspace(1)> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT: [[TMP2:%.*]] = insertelement <vscale x 4 x i64> poison, i64 [[ARG2]], {{i32|i64}} 0
; CHECK-NEXT: [[TMP3:%.*]] = shufflevector <vscale x 4 x i64> [[TMP2]], <vscale x 4 x i64> poison, <vscale x 4 x i32> zeroinitializer
; CHECK-NEXT: [[TMP4:%.*]] = call <vscale x 4 x i64> @llvm.{{(experimental\.)?}}stepvector.nxv4i64()
; CHECK-NEXT: [[TMP5:%.*]] = mul <vscale x 4 x i64> [[TMP3]], [[TMP4]]
; CHECK-NEXT: [[TMP6:%.*]] = getelementptr i32, <vscale x 4 x ptr addrspace(1)> [[TMP1]], <vscale x 4 x i64> [[TMP5]]
; CHECK-NEXT: call void @llvm.masked.scatter.nxv4i32.nxv4p1(<vscale x 4 x i32> [[ARG0]], <vscale x 4 x ptr addrspace(1)> [[TMP6]], i32 immarg 4, <vscale x 4 x i1> shufflevector (<vscale x 4 x i1> insertelement (<vscale x 4 x i1> poison, i1 true, {{i32|i64}} 0), <vscale x 4 x i1> poison, <vscale x 4 x i32> zeroinitializer)) [[MASKED_ATTRS:#[0-9]+]]
; CHECK-NEXT: ret void
; CHECK-NEXT: }

; CHECK-DAG: attributes [[ATTRS]] = { norecurse nounwind }
; CHECK-DAG: attributes [[MASKED_ATTRS]] = {
