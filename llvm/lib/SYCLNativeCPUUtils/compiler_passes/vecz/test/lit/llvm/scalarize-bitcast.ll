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

; RUN: veczc -vecz-simd-width=4 -vecz-passes=scalarize -vecz-choices=FullScalarization -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define dso_local spir_kernel void @bitcast1(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %gid = tail call i64 @__mux_get_global_id(i32 noundef 0)
  %pin = getelementptr inbounds <2 x float>, ptr addrspace(1) %in, i64 %gid
  %pout = getelementptr inbounds <4 x half>, ptr addrspace(1) %out, i64 %gid
  %0 = load <2 x float>, ptr addrspace(1) %pin, align 4
  %1 = bitcast <2 x float> %0 to <4 x half>
  store <4 x half> %1, ptr addrspace(1) %pout, align 4
  ret void
}

; CHECK-LABEL: define{{.*}}spir_kernel void @__vecz_v4_bitcast1
; CHECK:      [[A0:%.+]] = load float,
; CHECK-NEXT: [[C0:%.+]] = load float,
; CHECK-NEXT: [[A1:%.+]] = bitcast float [[A0]] to i32
; CHECK-NEXT: [[A2:%.+]] = trunc i32 [[A1]] to i16
; CHECK-NEXT: [[A3:%.+]] = bitcast i16 [[A2]] to half
; CHECK-NEXT: [[B1:%.+]] = bitcast float [[A0]] to i32
; CHECK-NEXT: [[B2:%.+]] = lshr i32 [[B1]], 16
; CHECK-NEXT: [[B3:%.+]] = trunc i32 [[B2]] to i16
; CHECK-NEXT: [[B4:%.+]] = bitcast i16 [[B3]] to half
; CHECK-NEXT: [[C1:%.+]] = bitcast float [[C0]] to i32
; CHECK-NEXT: [[C2:%.+]] = trunc i32 [[C1]] to i16
; CHECK-NEXT: [[C3:%.+]] = bitcast i16 [[C2]] to half
; CHECK-NEXT: [[D1:%.+]] = bitcast float [[C0]] to i32
; CHECK-NEXT: [[D2:%.+]] = lshr i32 [[D1]], 16
; CHECK-NEXT: [[D3:%.+]] = trunc i32 [[D2]] to i16
; CHECK-NEXT: [[D4:%.+]] = bitcast i16 [[D3]] to half
; CHECK:      store half [[A3]],
; CHECK-NEXT: store half [[B4]],
; CHECK-NEXT: store half [[C3]],
; CHECK-NEXT: store half [[D4]],
; CHECK-NEXT: ret void

define dso_local spir_kernel void @bitcast2(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %gid = tail call i64 @__mux_get_global_id(i32 noundef 0)
  %pin = getelementptr inbounds <4 x half>, ptr addrspace(1) %in, i64 %gid
  %pout = getelementptr inbounds <2 x float>, ptr addrspace(1) %out, i64 %gid
  %0 = load <4 x half>, ptr addrspace(1) %pin, align 4
  %1 = bitcast <4 x half> %0 to <2 x float>
  store <2 x float> %1, ptr addrspace(1) %pout, align 4
  ret void
}

; CHECK-LABEL: define{{.*}}spir_kernel void @__vecz_v4_bitcast2
; CHECK:      [[A0:%.+]] = load half,
; CHECK-NEXT: [[B0:%.+]] = load half,
; CHECK-NEXT: [[C0:%.+]] = load half,
; CHECK-NEXT: [[D0:%.+]] = load half,
; CHECK-NEXT: [[A1:%.+]] = bitcast half [[A0]] to i16
; CHECK-NEXT: [[A2:%.+]] = zext i16 [[A1]] to i32
; CHECK-NEXT: [[B1:%.+]] = bitcast half [[B0]] to i16
; CHECK-NEXT: [[B2:%.+]] = zext i16 [[B1]] to i32
; CHECK-NEXT: [[B3:%.+]] = shl i32 [[B2]], 16
; CHECK-NEXT: [[AB4:%.+]] = or i32 [[A2]], [[B3]]
; CHECK-NEXT: [[AB5:%.+]] = bitcast i32 [[AB4]] to float
; CHECK-NEXT: [[C1:%.+]] = bitcast half [[C0]] to i16
; CHECK-NEXT: [[C2:%.+]] = zext i16 [[C1]] to i32
; CHECK-NEXT: [[D1:%.+]] = bitcast half [[D0]] to i16
; CHECK-NEXT: [[D2:%.+]] = zext i16 [[D1]] to i32
; CHECK-NEXT: [[D3:%.+]] = shl i32 [[D2]], 16
; CHECK-NEXT: [[CD4:%.+]] = or i32 [[C2]], [[D3]]
; CHECK-NEXT: [[CD5:%.+]] = bitcast i32 [[CD4]] to float
; CHECK:      store float [[AB5]],
; CHECK-NEXT: store float [[CD5]],
; CHECK-NEXT: ret void

define dso_local spir_kernel void @bitcast3(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %gid = tail call i64 @__mux_get_global_id(i32 noundef 0)
  %pin = getelementptr inbounds <2 x i32>, ptr addrspace(1) %in, i64 %gid
  %pout = getelementptr inbounds <2 x float>, ptr addrspace(1) %out, i64 %gid
  %0 = load <2 x i32>, ptr addrspace(1) %pin, align 4
  %1 = bitcast <2 x i32> %0 to <2 x float>
  store <2 x float> %1, ptr addrspace(1) %pout, align 4
  ret void
}

; CHECK-LABEL: define{{.*}}spir_kernel void @__vecz_v4_bitcast3
; CHECK:      [[A0:%.+]] = load i32,
; CHECK-NEXT: [[B0:%.+]] = load i32,
; CHECK-NEXT: [[A1:%.+]] = bitcast i32 [[A0]] to float
; CHECK-NEXT: [[B1:%.+]] = bitcast i32 [[B0]] to float
; CHECK:      store float [[A1]],
; CHECK-NEXT: store float [[B1]],
; CHECK-NEXT: ret void

define dso_local spir_kernel void @bitcast4(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %gid = tail call i64 @__mux_get_global_id(i32 noundef 0)
  %pin = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %gid
  %pout = getelementptr inbounds <4 x i16>, ptr addrspace(1) %out, i64 %gid
  %0 = load i32, ptr addrspace(1) %pin, align 4
  %1 = insertelement <2 x i32> poison, i32 %0, i32 0
  %2 = bitcast <2 x i32> %1 to <4 x i16>
  %3 = shufflevector <4 x i16> %2, <4 x i16> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
  store <4 x i16> %3, ptr addrspace(1) %pout, align 4
  ret void
}

; CHECK-LABEL: define{{.*}}spir_kernel void @__vecz_v4_bitcast4
; CHECK:      [[A0:%.+]] = load i32,
; CHECK-NEXT: [[A1:%.+]] = trunc i32 [[A0]] to i16
; CHECK-NEXT: [[B0:%.+]] = lshr i32 %0, 16
; CHECK-NEXT: [[B1:%.+]] = trunc i32 [[B0]] to i16
; CHECK:      store i16 [[A1]],
; CHECK-NEXT: store i16 [[B1]],
; CHECK-NEXT: store i16 [[A1]],
; CHECK-NEXT: store i16 [[B1]],
; CHECK-NEXT: ret void

declare i64 @__mux_get_global_id(i32 noundef)
