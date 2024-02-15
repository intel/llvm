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

; RUN: veczc -w 4 -vecz-passes="print<strides>" -S < %s -o /dev/null 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK: StrideAnalysis for function '__vecz_v4_foo':
define spir_kernel void @foo(ptr addrspace(1) align 1 %input) {
entry:
  %localid0 = tail call i64 @__mux_get_local_id(i32 0)
  %localsize0 = tail call i64 @__mux_get_local_size(i32 0)
  %groupid0 = tail call i64 @__mux_get_group_id(i32 0)
  %globalid0 = tail call i64 @__mux_get_global_id(i32 0)

; CHECK: Stride for ptr addrspace(1) %input
; CHECK-NEXT: uniform
  %lduniform = load i8, ptr addrspace(1) %input, align 1

; CHECK: Stride for %arrayidx0 = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %globalid0
; CHECK-NEXT: linear stride of 1
  %arrayidx0 = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %globalid0
  %ld0 = load i8, ptr addrspace(1) %arrayidx0, align 1

  %truncglobalid0 = trunc i64 %globalid0 to i32

; CHECK: Stride for %arrayidx1 = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %sexttruncglobalid0
; CHECK-NEXT: linear stride of 1
  %sexttruncglobalid0 = sext i32 %truncglobalid0 to i64
  %arrayidx1 = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %sexttruncglobalid0
  %ld1 = load i8, ptr addrspace(1) %arrayidx1, align 1

; CHECK: Stride for %arrayidx2 = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %zexttruncglobalid0
; CHECK-NEXT: divergent
  %zexttruncglobalid0 = zext i32 %truncglobalid0 to i64
  %arrayidx2 = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %zexttruncglobalid0
  %ld2 = load i8, ptr addrspace(1) %arrayidx2, align 1

; CHECK: Stride for %arrayidx3 = getelementptr inbounds i32, ptr addrspace(1) %input, i64 %globalid0
; CHECK-NEXT: linear stride of 4
  %arrayidx3 = getelementptr inbounds i32, ptr addrspace(1) %input, i64 %globalid0
  %ld3 = load i8, ptr addrspace(1) %arrayidx3, align 1

; CHECK: Stride for %arrayidx4 = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %globalid0mul8
; CHECK-NEXT: linear stride of 8
  %globalid0mul8 = mul i64 %globalid0, 8
  %arrayidx4 = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %globalid0mul8
  %ld4 = load i8, ptr addrspace(1) %arrayidx4, align 1

; CHECK: Stride for %arrayidx5 = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %globalid0mul16
; CHECK-NEXT: linear stride of 16
  %globalid0mul16 = mul i64 %globalid0mul8, 2
  %arrayidx5 = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %globalid0mul16
  %ld5 = load i8, ptr addrspace(1) %arrayidx5, align 1

; CHECK: Stride for %arrayidx6 = getelementptr inbounds i32, ptr addrspace(1) %input, i64 %globalid0mul8
; CHECK-NEXT: linear stride of 32
  %arrayidx6 = getelementptr inbounds i32, ptr addrspace(1) %input, i64 %globalid0mul8
  %ld6 = load i32, ptr addrspace(1) %arrayidx6, align 1

; CHECK: Stride for %arrayidx7 = getelementptr inbounds i16, ptr addrspace(1) %input, i64 %idxprom7
; CHECK-NEXT: linear stride of 2
  %mul7 = mul i64 %localsize0, %groupid0
  %add7 = add i64 %mul7, %localid0
  %trunc7 = trunc i64 %add7 to i32
  %conv7 = add i32 %trunc7, -1
  %idxprom7 = sext i32 %conv7 to i64
  %arrayidx7 = getelementptr inbounds i16, ptr addrspace(1) %input, i64 %idxprom7
  %ld7 = load i16, ptr addrspace(1) %arrayidx7, align 1

; CHECK: Stride for %arrayidx8 = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %idxprom8
; CHECK-NEXT: divergent
  %mul8 = mul i64 %localsize0, %groupid0
  %add8 = add i64 %mul8, %localid0
  %trunc8 = trunc i64 %add8 to i32
  %conv8 = add i32 %trunc8, -1
  %idxprom8 = zext i32 %conv8 to i64
  %arrayidx8 = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %idxprom8
  %ld8 = load i8, ptr addrspace(1) %arrayidx8, align 1

; CHECK: Stride for %arrayidx9 = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %idxprom9
; CHECK-NEXT: divergent
  %mul9 = mul i64 %groupid0, %localsize0
  %add9 = add nuw nsw i64 %localid0, 4294967295
  %conv9 = add i64 %add9, %mul9
  %idxprom9 = and i64 %conv9, 4294967295
  %arrayidx9 = getelementptr inbounds i8, ptr addrspace(1) %input, i64 %idxprom9
  %ld9 = load i8, ptr addrspace(1) %arrayidx9, align 1

  ret void
}

declare i64 @__mux_get_local_id(i32)
declare i64 @__mux_get_local_size(i32)
declare i64 @__mux_get_group_id(i32)
declare i64 @__mux_get_global_id(i32)
