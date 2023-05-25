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

; RUN: veczc -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

%struct.PerItemKernelInfo = type <{ <4 x i64>, i32, i32 }>

; Function start
; CHECK: spir_kernel void @__vecz_v4_foo(

; There should be exactly 4 vector stores
; CHECK: store <4 x i64>
; CHECK: store <4 x i64>
; CHECK: store <4 x i64>
; CHECK: store <4 x i64>
; CHECK-NOT: call void @__vecz_b_scatter_store1_Dv4_mDv4_{{.*}}(<4 x i64>
; CHECK-NOT: call void @__vecz_b_interleaved_store1_5_Dv4_{{.*}}(<4 x i64>
 
; There is one interleaved store from the scalar write
; CHECK: call void @__vecz_b_interleaved_store1_10_Dv4_j{{(u3ptrU3AS1|PU3AS1j)}}(<4 x i32>
 
; There shouldn't be any other stores
; CHECK-NOT: call void @__vecz_b_{{.*}}_store
 
; Function end
; CHECK: ret void

define dso_local spir_kernel void @foo(%struct.PerItemKernelInfo addrspace(1)* nocapture noundef writeonly %info) !reqd_work_group_size !11 {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 noundef 0)
  %call1 = tail call spir_func i64 @_Z13get_global_idj(i32 noundef 1)
  %call2 = tail call spir_func i64 @_Z13get_global_idj(i32 noundef 2)
  %call3 = tail call spir_func i64 @_Z15get_global_sizej(i32 noundef 0)
  %call5 = tail call spir_func i64 @_Z15get_global_sizej(i32 noundef 1)
  %mul7 = mul nuw nsw i64 %call5, %call2
  %reass.add = add nuw nsw i64 %mul7, %call1
  %reass.mul = mul nuw nsw i64 %reass.add, %call3
  %add8 = add nuw nsw i64 %reass.mul, %call
  %vecinit = insertelement <4 x i64> undef, i64 %call3, i64 0
  %vecinit11 = insertelement <4 x i64> %vecinit, i64 %call5, i64 1
  %call12 = tail call spir_func i64 @_Z15get_global_sizej(i32 noundef 2)
  %vecinit13 = insertelement <4 x i64> %vecinit11, i64 %call12, i64 2
  %call14 = tail call spir_func i64 @_Z15get_global_sizej(i32 noundef 3)
  %vecinit15 = insertelement <4 x i64> %vecinit13, i64 %call14, i64 3
  %global_size = getelementptr inbounds %struct.PerItemKernelInfo, %struct.PerItemKernelInfo addrspace(1)* %info, i64 %add8, i32 0
  store <4 x i64> %vecinit15, <4 x i64> addrspace(1)* %global_size, align 1
  %call16 = tail call spir_func i32 @_Z12get_work_dimv()
  %work_dim = getelementptr inbounds %struct.PerItemKernelInfo, %struct.PerItemKernelInfo addrspace(1)* %info, i64 %add8, i32 1
  store i32 %call16, i32 addrspace(1)* %work_dim, align 1
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)

declare spir_func i64 @_Z15get_global_sizej(i32)

declare spir_func i32 @_Z12get_work_dimv()

!11 = !{i32 4, i32 1, i32 1}

