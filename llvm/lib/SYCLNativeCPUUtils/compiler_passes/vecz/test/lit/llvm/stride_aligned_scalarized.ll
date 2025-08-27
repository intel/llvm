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

; RUN: veczc -S < %s -vecz-choices=FullScalarization | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-p:64:64:64-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.PerItemKernelInfo = type <{ <4 x i64>, i32, i32 }>

define spir_kernel void @foo(%struct.PerItemKernelInfo addrspace(1)* %info) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 0)
  %call1 = tail call i64 @__mux_get_global_id(i32 1)
  %call2 = tail call i64 @__mux_get_global_id(i32 2)
  %call3 = tail call i64 @__mux_get_global_size(i32 0)
  %call5 = tail call i64 @__mux_get_global_size(i32 1)
  %mul7 = mul nuw nsw i64 %call5, %call2
  %reass.add = add nuw nsw i64 %mul7, %call1
  %reass.mul = mul nuw nsw i64 %reass.add, %call3
  %add8 = add nuw nsw i64 %reass.mul, %call
  %vecinit = insertelement <4 x i64> poison, i64 %call3, i64 0
  %vecinit11 = insertelement <4 x i64> %vecinit, i64 %call5, i64 1
  %call12 = tail call i64 @__mux_get_global_size(i32 2)
  %vecinit13 = insertelement <4 x i64> %vecinit11, i64 %call12, i64 2
  %call14 = tail call i64 @__mux_get_global_size(i32 3)
  %vecinit15 = insertelement <4 x i64> %vecinit13, i64 %call14, i64 3
  %global_size = getelementptr inbounds %struct.PerItemKernelInfo, %struct.PerItemKernelInfo addrspace(1)* %info, i64 %add8, i32 0
  store <4 x i64> %vecinit15, <4 x i64> addrspace(1)* %global_size, align 1
  %call16 = tail call i32 @__mux_get_work_dim()
  %work_dim = getelementptr inbounds %struct.PerItemKernelInfo, %struct.PerItemKernelInfo addrspace(1)* %info, i64 %add8, i32 1
  store i32 %call16, i32 addrspace(1)* %work_dim, align 1
  ret void
}

declare i64 @__mux_get_global_id(i32)

declare i64 @__mux_get_global_size(i32)

declare i32 @__mux_get_work_dim()

; CHECK: spir_kernel void @foo
; CHECK: call void @__vecz_b_interleaved_store1_5_Dv4_m{{(u3ptrU3AS1|PU3AS1m)}}(<4 x i64>
; CHECK: call void @__vecz_b_interleaved_store1_5_Dv4_m{{(u3ptrU3AS1|PU3AS1m)}}(<4 x i64>
; CHECK: call void @__vecz_b_interleaved_store1_5_Dv4_m{{(u3ptrU3AS1|PU3AS1m)}}(<4 x i64>
; CHECK: call void @__vecz_b_interleaved_store1_5_Dv4_m{{(u3ptrU3AS1|PU3AS1m)}}(<4 x i64>
; CHECK-NOT: call void @__vecz_b_interleaved_store1_5_Dv4_m{{.*}}(<4 x i64>
; CHECK: call void @__vecz_b_interleaved_store1_10_Dv4_j{{(u3ptrU3AS1|PU3AS1j)}}(<4 x i32>
; CHECK-NOT: call void @__vecz_b_{{.*}}_store
; CHECK: ret void
