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

; REQUIRES: llvm-13+
; RUN: veczc -k store_element -vecz-target-triple="riscv64-unknown-unknown" -vecz-target-features=+f,+d,%vattr -vecz-simd-width=4 -vecz-scalable -vecz-choices=VectorPredication -S < %s | FileCheck %s --check-prefix CHECK-STORE-4
; RUN: veczc -k store_element -vecz-target-triple="riscv64-unknown-unknown" -vecz-target-features=+f,+d,%vattr -vecz-simd-width=8 -vecz-scalable -vecz-choices=VectorPredication -S < %s | FileCheck %s --check-prefix CHECK-STORE-8
; RUN: veczc -k store_element -vecz-target-triple="riscv64-unknown-unknown" -vecz-target-features=+f,+d,%vattr -vecz-simd-width=16 -vecz-scalable -vecz-choices=VectorPredication -S < %s | FileCheck %s --check-prefix CHECK-STORE-16
; RUN: veczc -k load_element -vecz-target-triple="riscv64-unknown-unknown" -vecz-target-features=+f,+d,%vattr -vecz-simd-width=4 -vecz-scalable -vecz-choices=VectorPredication -S < %s | FileCheck %s --check-prefix CHECK-LOAD-4
; RUN: veczc -k load_element -vecz-target-triple="riscv64-unknown-unknown" -vecz-target-features=+f,+d,%vattr -vecz-simd-width=8 -vecz-scalable -vecz-choices=VectorPredication -S < %s | FileCheck %s --check-prefix CHECK-LOAD-8
; RUN: veczc -k load_element -vecz-target-triple="riscv64-unknown-unknown" -vecz-target-features=+f,+d,%vattr -vecz-simd-width=16 -vecz-scalable -vecz-choices=VectorPredication -S < %s | FileCheck %s --check-prefix CHECK-LOAD-16

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @store_element(i32 %0, i32 addrspace(1)* %b) {
entry:
  %call = call i64 @__mux_get_global_id(i32 0)
  %cond = icmp ne i64 %call, 0
  br i1 %cond, label %do, label %ret

do:
  %dest = getelementptr inbounds i32, i32 addrspace(1)* %b, i64 %call
  store i32 %0, i32 addrspace(1)* %dest, align 4
  br label %ret

ret:
  ret void
}

; CHECK-STORE-4:       define void @__vecz_b_masked_store4_vp_u5nxv4ju3ptrU3AS1u5nxv4bj(<vscale x 4 x i32> [[TMP0:%.*]], ptr addrspace(1) [[TMP1:%.*]], <vscale x 4 x i1> [[TMP2:%.*]], i32 [[TMP3:%.*]]) {
; CHECK-STORE-4-NEXT:  entry:
; CHECK-STORE-4-NEXT:    call void @llvm.vp.store.nxv4i32.p1(<vscale x 4 x i32> [[TMP0]], ptr addrspace(1) [[TMP1]], <vscale x 4 x i1> [[TMP2]], i32 [[TMP3]])
; CHECK-STORE-4-NEXT:    ret void

; CHECK-STORE-8:       define void @__vecz_b_masked_store4_vp_u5nxv8ju3ptrU3AS1u5nxv8bj(<vscale x 8 x i32> [[TMP0:%.*]], ptr addrspace(1) [[TMP1:%.*]], <vscale x 8 x i1> [[TMP2:%.*]], i32 [[TMP3:%.*]]) {
; CHECK-STORE-8-NEXT:  entry:
; CHECK-STORE-8-NEXT:    call void @llvm.vp.store.nxv8i32.p1(<vscale x 8 x i32> [[TMP0]], ptr addrspace(1) [[TMP1]], <vscale x 8 x i1> [[TMP2]], i32 [[TMP3]])
; CHECK-STORE-8-NEXT:    ret void

; CHECK-STORE-16:       define void @__vecz_b_masked_store4_vp_u6nxv16ju3ptrU3AS1u6nxv16bj(<vscale x 16 x i32> [[TMP0:%.*]], ptr addrspace(1) [[TMP1:%.*]], <vscale x 16 x i1> [[TMP2:%.*]], i32 [[TMP3:%.*]]) {
; CHECK-STORE-16-NEXT:  entry:
; CHECK-STORE-16-NEXT:    [[TMP5:%.*]] = call <vscale x 16 x i32> @llvm.experimental.stepvector.nxv16i32()
; CHECK-STORE-16-NEXT:    [[SPLATINSERT:%.*]] = insertelement <vscale x 16 x i32> poison, i32 [[TMP3]], {{i32|i64}} 0
; CHECK-STORE-16-NEXT:    [[SPLAT:%.*]] = shufflevector <vscale x 16 x i32> [[SPLATINSERT]], <vscale x 16 x i32> poison, <vscale x 16 x i32> zeroinitializer
; CHECK-STORE-16-NEXT:    [[TMP6:%.*]] = icmp ult <vscale x 16 x i32> [[TMP5]], [[SPLAT]]
; CHECK-STORE-16-NEXT:    [[TMP7:%.*]] = select <vscale x 16 x i1> [[TMP2]], <vscale x 16 x i1> [[TMP6]], <vscale x 16 x i1> zeroinitializer
; CHECK-STORE-16-NEXT:    call void @llvm.masked.store.nxv16i32.p1(<vscale x 16 x i32> [[TMP0]], ptr addrspace(1) [[TMP1]], i32 4, <vscale x 16 x i1> [[TMP7]])
; CHECK-STORE-16-NEXT:    ret void

define spir_kernel void @load_element(i32 addrspace(1)* %a, i32 addrspace(1)* %b) {
entry:
  %call = call i64 @__mux_get_global_id(i32 0)
  %cond = icmp ne i64 %call, 0
  br i1 %cond, label %do, label %ret

do:
  %src = getelementptr inbounds i32, i32 addrspace(1)* %a, i64 %call
  %dest = getelementptr inbounds i32, i32 addrspace(1)* %b, i64 %call
  %do.res = load i32, i32 addrspace(1)* %src, align 4
  store i32 %do.res, i32 addrspace(1)* %dest, align 4
  br label %ret

ret:
  ret void
}

; CHECK-LOAD-4:      define <vscale x 4 x i32> @__vecz_b_masked_load4_vp_u5nxv4ju3ptrU3AS1u5nxv4bj(ptr addrspace(1) [[TMP0:%.*]], <vscale x 4 x i1> [[TMP1:%.*]], i32 [[TMP2:%.*]]) {
; CHECK-LOAD-4-NEXT: entry:
; CHECK-LOAD-4-NEXT:   [[TMP4:%.*]] = call <vscale x 4 x i32> @llvm.vp.load.nxv4i32.p1(ptr addrspace(1) [[TMP0]], <vscale x 4 x i1> [[TMP1]], i32 [[TMP2]])
; CHECK-LOAD-4-NEXT:   ret <vscale x 4 x i32> [[TMP4]]

; CHECK-LOAD-8:      define <vscale x 8 x i32> @__vecz_b_masked_load4_vp_u5nxv8ju3ptrU3AS1u5nxv8bj(ptr addrspace(1) [[TMP0:%.*]], <vscale x 8 x i1> [[TMP1:%.*]], i32 [[TMP2:%.*]]) {
; CHECK-LOAD-8-NEXT: entry:
; CHECK-LOAD-8-NEXT:   [[TMP4:%.*]] = call <vscale x 8 x i32> @llvm.vp.load.nxv8i32.p1(ptr addrspace(1) [[TMP0]], <vscale x 8 x i1> [[TMP1]], i32 [[TMP2]])
; CHECK-LOAD-8-NEXT:   ret <vscale x 8 x i32> [[TMP4]]

; CHECK-LOAD-16:      define <vscale x 16 x i32> @__vecz_b_masked_load4_vp_u6nxv16ju3ptrU3AS1u6nxv16bj(ptr addrspace(1) [[TMP0:%.*]], <vscale x 16 x i1> [[TMP1:%.*]], i32 [[TMP2:%.*]]) {
; CHECK-LOAD-16-NEXT: entry:
; CHECK-LOAD-16-NEXT: [[TMP4:%.*]] = call <vscale x 16 x i32> @llvm.experimental.stepvector.nxv16i32()
; CHECK-LOAD-16-NEXT: [[TMPSPLATINSERT:%.*]] = insertelement <vscale x 16 x i32> poison, i32 [[TMP2]], {{i32|i64}} 0
; CHECK-LOAD-16-NEXT: [[TMPSPLAT:%.*]] = shufflevector <vscale x 16 x i32> [[TMPSPLATINSERT]], <vscale x 16 x i32> poison, <vscale x 16 x i32> zeroinitializer
; CHECK-LOAD-16-NEXT: [[TMP5:%.*]] = icmp ult <vscale x 16 x i32> [[TMP4]], [[TMPSPLAT]]
; CHECK-LOAD-16-NEXT: [[TMP6:%.*]] = select <vscale x 16 x i1> [[TMP1]], <vscale x 16 x i1> [[TMP5]], <vscale x 16 x i1> zeroinitializer
; CHECK-LOAD-16-NEXT: [[TMP7:%.*]] = call <vscale x 16 x i32> @llvm.masked.load.nxv16i32.p1(ptr addrspace(1) [[TMP0]], i32 4, <vscale x 16 x i1> [[TMP6]], <vscale x 16 x i32> poison)
; CHECK-LOAD-16-NEXT: ret <vscale x 16 x i32> [[TMP7]]

declare i64 @__mux_get_global_id(i32)
