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

; RUN: veczc -k test_instructions -vecz-passes=scalarize -vecz-simd-width=4 -vecz-choices=FullScalarization -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare i64 @__mux_get_global_id(i32)

define spir_kernel void @test_instructions(<4 x i32>* %pa, <4 x i32>* %pb, <4 x i32>* %pc) {
entry:
  %idx = call i64 @__mux_get_global_id(i32 0)
  %a = getelementptr <4 x i32>, <4 x i32>* %pa, i64 %idx
  %b = getelementptr <4 x i32>, <4 x i32>* %pb, i64 %idx
  %c = getelementptr <4 x i32>, <4 x i32>* %pc, i64 %idx
  %0 = load <4 x i32>, <4 x i32>* %a, align 16
  %1 = load <4 x i32>, <4 x i32>* %b, align 16
  %add = add <4 x i32> %1, %0
  store <4 x i32> %add, <4 x i32>* %c, align 16
  %arrayidx3 = getelementptr inbounds <4 x i32>, <4 x i32>* %a, i64 1
  %2 = load <4 x i32>, <4 x i32>* %arrayidx3, align 16
  %arrayidx4 = getelementptr inbounds <4 x i32>, <4 x i32>* %b, i64 1
  %3 = load <4 x i32>, <4 x i32>* %arrayidx4, align 16
  %cmp = icmp sgt <4 x i32> %2, %3
  %sext = sext <4 x i1> %cmp to <4 x i32>
  %arrayidx5 = getelementptr inbounds <4 x i32>, <4 x i32>* %c, i64 1
  store <4 x i32> %sext, <4 x i32>* %arrayidx5, align 16
  %arrayidx6 = getelementptr inbounds <4 x i32>, <4 x i32>* %a, i64 2
  %4 = load <4 x i32>, <4 x i32>* %arrayidx6, align 16
  %cmp7 = icmp slt <4 x i32> %4, <i32 11, i32 12, i32 13, i32 14>
  %sext8 = sext <4 x i1> %cmp7 to <4 x i32>
  %arrayidx9 = getelementptr inbounds <4 x i32>, <4 x i32>* %c, i64 2
  store <4 x i32> %sext8, <4 x i32>* %arrayidx9, align 16
  ret void
}

; CHECK: define spir_kernel void @__vecz_v4_test_instructions(ptr %pa, ptr %pb, ptr %pc)
; CHECK: entry:
; CHECK: %[[A_0:.+]] = getelementptr i32, ptr %a, i32 0
; CHECK: %[[A_1:.+]] = getelementptr i32, ptr %a, i32 1
; CHECK: %[[A_2:.+]] = getelementptr i32, ptr %a, i32 2
; CHECK: %[[A_3:.+]] = getelementptr i32, ptr %a, i32 3
; CHECK: %[[LA_0:.+]] = load i32, ptr %[[A_0]]
; CHECK: %[[LA_1:.+]] = load i32, ptr %[[A_1]]
; CHECK: %[[LA_2:.+]] = load i32, ptr %[[A_2]]
; CHECK: %[[LA_3:.+]] = load i32, ptr %[[A_3]]
; CHECK: %[[B_0:.+]] = getelementptr i32, ptr %b, i32 0
; CHECK: %[[B_1:.+]] = getelementptr i32, ptr %b, i32 1
; CHECK: %[[B_2:.+]] = getelementptr i32, ptr %b, i32 2
; CHECK: %[[B_3:.+]] = getelementptr i32, ptr %b, i32 3
; CHECK: %[[LB_0:.+]] = load i32, ptr %[[B_0]]
; CHECK: %[[LB_1:.+]] = load i32, ptr %[[B_1]]
; CHECK: %[[LB_2:.+]] = load i32, ptr %[[B_2]]
; CHECK: %[[LB_3:.+]] = load i32, ptr %[[B_3]]
; CHECK: %[[ADD1:.+]] = add i32 %[[LB_0]], %[[LA_0]]
; CHECK: %[[ADD2:.+]] = add i32 %[[LB_1]], %[[LA_1]]
; CHECK: %[[ADD3:.+]] = add i32 %[[LB_2]], %[[LA_2]]
; CHECK: %[[ADD4:.+]] = add i32 %[[LB_3]], %[[LA_3]]
; CHECK: %[[C_0:.+]] = getelementptr i32, ptr %c, i32 0
; CHECK: %[[C_1:.+]] = getelementptr i32, ptr %c, i32 1
; CHECK: %[[C_2:.+]] = getelementptr i32, ptr %c, i32 2
; CHECK: %[[C_3:.+]] = getelementptr i32, ptr %c, i32 3
; CHECK: store i32 %[[ADD1]], ptr %[[C_0]]
; CHECK: store i32 %[[ADD2]], ptr %[[C_1]]
; CHECK: store i32 %[[ADD3]], ptr %[[C_2]]
; CHECK: store i32 %[[ADD4]], ptr %[[C_3]]
; CHECK: %arrayidx3 = getelementptr <4 x i32>, ptr %a, i64 1
; CHECK: %[[A1_0:.+]] = getelementptr i32, ptr %arrayidx3, i32 0
; CHECK: %[[A1_1:.+]] = getelementptr i32, ptr %arrayidx3, i32 1
; CHECK: %[[A1_2:.+]] = getelementptr i32, ptr %arrayidx3, i32 2
; CHECK: %[[A1_3:.+]] = getelementptr i32, ptr %arrayidx3, i32 3
; CHECK: %[[LA1_0:.+]] = load i32, ptr %[[A1_0]]
; CHECK: %[[LA1_1:.+]] = load i32, ptr %[[A1_1]]
; CHECK: %[[LA1_2:.+]] = load i32, ptr %[[A1_2]]
; CHECK: %[[LA1_3:.+]] = load i32, ptr %[[A1_3]]
; CHECK: %arrayidx4 = getelementptr <4 x i32>, ptr %b, i64 1
; CHECK: %[[B1_0:.+]] = getelementptr i32, ptr %arrayidx4, i32 0
; CHECK: %[[B1_1:.+]] = getelementptr i32, ptr %arrayidx4, i32 1
; CHECK: %[[B1_2:.+]] = getelementptr i32, ptr %arrayidx4, i32 2
; CHECK: %[[B1_3:.+]] = getelementptr i32, ptr %arrayidx4, i32 3
; CHECK: %[[LB1_0:.+]] = load i32, ptr %[[B1_0]]
; CHECK: %[[LB1_1:.+]] = load i32, ptr %[[B1_1]]
; CHECK: %[[LB1_2:.+]] = load i32, ptr %[[B1_2]]
; CHECK: %[[LB1_3:.+]] = load i32, ptr %[[B1_3]]
; CHECK: %[[CMP5:.+]] = icmp sgt i32 %[[LA1_0]], %[[LB1_0]]
; CHECK: %[[CMP6:.+]] = icmp sgt i32 %[[LA1_1]], %[[LB1_1]]
; CHECK: %[[CMP8:.+]] = icmp sgt i32 %[[LA1_2]], %[[LB1_2]]
; CHECK: %[[CMP9:.+]] = icmp sgt i32 %[[LA1_3]], %[[LB1_3]]
; CHECK: %[[SEXT10:.+]] = sext i1 %[[CMP5]] to i32
; CHECK: %[[SEXT11:.+]] = sext i1 %[[CMP6]] to i32
; CHECK: %[[SEXT12:.+]] = sext i1 %[[CMP8]] to i32
; CHECK: %[[SEXT13:.+]] = sext i1 %[[CMP9]] to i32
; CHECK: %arrayidx5 = getelementptr <4 x i32>, ptr %c, i64 1
; CHECK: %[[C1_0:.+]] = getelementptr i32, ptr %arrayidx5, i32 0
; CHECK: %[[C1_1:.+]] = getelementptr i32, ptr %arrayidx5, i32 1
; CHECK: %[[C1_2:.+]] = getelementptr i32, ptr %arrayidx5, i32 2
; CHECK: %[[C1_3:.+]] = getelementptr i32, ptr %arrayidx5, i32 3
; CHECK: store i32 %[[SEXT10]], ptr %[[C1_0]]
; CHECK: store i32 %[[SEXT11]], ptr %[[C1_1]]
; CHECK: store i32 %[[SEXT12]], ptr %[[C1_2]]
; CHECK: store i32 %[[SEXT13]], ptr %[[C1_3]]
; CHECK: %arrayidx6 = getelementptr <4 x i32>, ptr %a, i64 2
; CHECK: %[[A2_0:.+]] = getelementptr i32, ptr %arrayidx6, i32 0
; CHECK: %[[A2_1:.+]] = getelementptr i32, ptr %arrayidx6, i32 1
; CHECK: %[[A2_2:.+]] = getelementptr i32, ptr %arrayidx6, i32 2
; CHECK: %[[A2_3:.+]] = getelementptr i32, ptr %arrayidx6, i32 3
; CHECK: %[[LA2_0:.+]] = load i32, ptr %[[A2_0]]
; CHECK: %[[LA2_1:.+]] = load i32, ptr %[[A2_1]]
; CHECK: %[[LA2_2:.+]] = load i32, ptr %[[A2_2]]
; CHECK: %[[LA2_3:.+]] = load i32, ptr %[[A2_3]]
; CHECK: %[[CMP714:.+]] = icmp slt i32 %[[LA2_0]], 11
; CHECK: %[[CMP715:.+]] = icmp slt i32 %[[LA2_1]], 12
; CHECK: %[[CMP716:.+]] = icmp slt i32 %[[LA2_2]], 13
; CHECK: %[[CMP717:.+]] = icmp slt i32 %[[LA2_3]], 14
; CHECK: %[[SEXT818:.+]] = sext i1 %[[CMP714]] to i32
; CHECK: %[[SEXT819:.+]] = sext i1 %[[CMP715]] to i32
; CHECK: %[[SEXT820:.+]] = sext i1 %[[CMP716]] to i32
; CHECK: %[[SEXT821:.+]] = sext i1 %[[CMP717]] to i32
; CHECK: %arrayidx9 = getelementptr <4 x i32>, ptr %c, i64 2
; CHECK: %[[C2_0:.+]] = getelementptr i32, ptr %arrayidx9, i32 0
; CHECK: %[[C2_1:.+]] = getelementptr i32, ptr %arrayidx9, i32 1
; CHECK: %[[C2_2:.+]] = getelementptr i32, ptr %arrayidx9, i32 2
; CHECK: %[[C2_3:.+]] = getelementptr i32, ptr %arrayidx9, i32 3
; CHECK: store i32 %[[SEXT818]], ptr %[[C2_0]]
; CHECK: store i32 %[[SEXT819]], ptr %[[C2_1]]
; CHECK: store i32 %[[SEXT820]], ptr %[[C2_2]]
; CHECK: store i32 %[[SEXT821]], ptr %[[C2_3]]
; CHECK: ret void
