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

; RUN: veczc -k gep -vecz-simd-width=4 -vecz-passes=scalarize -vecz-choices=FullScalarization -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define dso_local spir_kernel void @gep(ptr addrspace(1) %data, ptr addrspace(1) %out) {
entry:
  %call = tail call i64 @__mux_get_global_id(i32 noundef 0)
  %ptrdata = getelementptr inbounds <2 x ptr>, ptr addrspace(1) %data, i64 %call
  %ptrdatavec = load <2 x ptr addrspace(1)>, ptr addrspace(1) %ptrdata
  %ptrdatavec.gep = getelementptr inbounds i32, <2 x ptr addrspace(1)> %ptrdatavec, i64 1
  %vec1 = call <2 x i32> @llvm.masked.gather.v2i32.v2p1(<2 x ptr addrspace(1)> %ptrdatavec, i32 16, <2 x i1> zeroinitializer, <2 x ptr addrspace(1)> zeroinitializer)
  %vec2 = call <2 x i32> @llvm.masked.gather.v2i32.v2p1(<2 x ptr addrspace(1)> %ptrdatavec.gep, i32 16, <2 x i1> zeroinitializer, <2 x ptr addrspace(1)> zeroinitializer)
  %vec.add = add <2 x i32> %vec1, %vec2
  %ptrout = getelementptr inbounds <2 x i32>, ptr addrspace(1) %out, i64 %call
  store <2 x i32> %vec.add, ptr addrspace(1) %ptrout
  ret void
}

declare i64 @__mux_get_global_id(i32 noundef)

declare <2 x i32> @llvm.masked.gather.v2i32.v2p1(<2 x ptr addrspace(1)>, i32, <2 x i1>, <2 x ptr addrspace(1)>)

; The full scalarization has not completely removed the vectors, the gather
; operation should have been replaced by non-vector loads, but check that at
; least we do not crash.

; CHECK: void @__vecz_v4_gep({{.*}})
; CHECK: entry:
; CHECK:   %call = tail call i64 @__mux_get_global_id(i32 noundef 0)
; CHECK:   %ptrdata = getelementptr <2 x ptr>, ptr addrspace(1) %data, i64 %call
; CHECK:   %0 = getelementptr ptr addrspace(1), ptr addrspace(1) %ptrdata, i32 0
; CHECK:   %1 = getelementptr ptr addrspace(1), ptr addrspace(1) %ptrdata, i32 1
; CHECK:   %ptrdatavec1 = load ptr addrspace(1), ptr addrspace(1) %0, align 1
; CHECK:   %ptrdatavec2 = load ptr addrspace(1), ptr addrspace(1) %1, align 1
; CHECK:   %2 = insertelement <2 x ptr addrspace(1)> undef, ptr addrspace(1) %ptrdatavec1, i32 0
; CHECK:   %3 = insertelement <2 x ptr addrspace(1)> %2, ptr addrspace(1) %ptrdatavec2, i32 1
; CHECK:   %ptrdatavec.gep3 = getelementptr i32, ptr addrspace(1) %ptrdatavec1, i64 1
; CHECK:   %ptrdatavec.gep4 = getelementptr i32, ptr addrspace(1) %ptrdatavec2, i64 1
; CHECK:   %4 = insertelement <2 x ptr addrspace(1)> undef, ptr addrspace(1) %ptrdatavec.gep3, i32 0
; CHECK:   %5 = insertelement <2 x ptr addrspace(1)> %4, ptr addrspace(1) %ptrdatavec.gep4, i32 1
; CHECK:   %vec1 = call <2 x i32> @llvm.masked.gather.v2i32.v2p1(<2 x ptr addrspace(1)> %3, i32 16, <2 x i1> zeroinitializer, <2 x ptr addrspace(1)> zeroinitializer)
; CHECK:   %6 = extractelement <2 x i32> %vec1, i32 0
; CHECK:   %7 = extractelement <2 x i32> %vec1, i32 1
; CHECK:   %vec2 = call <2 x i32> @llvm.masked.gather.v2i32.v2p1(<2 x ptr addrspace(1)> %5, i32 16, <2 x i1> zeroinitializer, <2 x ptr addrspace(1)> zeroinitializer)
; CHECK:   %8 = extractelement <2 x i32> %vec2, i32 0
; CHECK:   %9 = extractelement <2 x i32> %vec2, i32 1
; CHECK:   %vec.add5 = add i32 %6, %8
; CHECK:   %vec.add6 = add i32 %7, %9
; CHECK:   %ptrout = getelementptr <2 x i32>, ptr addrspace(1) %out, i64 %call
; CHECK:   %10 = getelementptr i32, ptr addrspace(1) %ptrout, i32 0
; CHECK:   %11 = getelementptr i32, ptr addrspace(1) %ptrout, i32 1
; CHECK:   store i32 %vec.add5, ptr addrspace(1) %10, align 4
; CHECK:   store i32 %vec.add6, ptr addrspace(1) %11, align 4
; CHECK:   ret void
