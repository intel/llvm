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

; RUN: veczc -vecz-passes=squash-small-vecs -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; It checks that the <4 x i8> is converted into a i32 and uses shifts and masks
; to implement the extract elements and zexts.
; CHECK: void @__vecz_v4_squashv4i8(
; CHECK:  %[[DATA:.+]] = load <4 x i8>
; CHECK:  %[[FREEZE:.+]] = freeze <4 x i8> %[[DATA]]
; CHECK:  %[[SQUASH:.+]] = bitcast <4 x i8> %[[FREEZE]] to i32
; CHECK:  %[[ZEXT0:.+]] = and i32 %[[SQUASH]], 255
; CHECK:  %[[EXTR1:.+]] = lshr i32 %[[SQUASH]], 8
; CHECK:  %[[ZEXT1:.+]] = and i32 %[[EXTR1]], 255
; CHECK:  %[[EXTR2:.+]] = lshr i32 %[[SQUASH]], 16
; CHECK:  %[[ZEXT2:.+]] = and i32 %[[EXTR2]], 255
; CHECK:  %[[EXTR3:.+]] = lshr i32 %[[SQUASH]], 24
; CHECK:  %[[ZEXT3:.+]] = and i32 %[[EXTR3]], 255
; CHECK:  %[[SUM1:.+]] = add i32 %[[ZEXT0]], %[[ZEXT1]]
; CHECK:  %[[SUM2:.+]] = xor i32 %[[SUM1]], %[[ZEXT2]]
; CHECK:  %[[SUM3:.+]] = and i32 %[[SUM2]], %[[ZEXT3]]
; CHECK:  ret void
define spir_kernel void @squashv4i8(ptr addrspace(1) %data, ptr addrspace(1) %output) #0 {
entry:
  %gid = call i64 @__mux_get_global_id(i64 0) #1
  %data.ptr = getelementptr inbounds <4 x i8>, ptr addrspace(1) %data, i64 %gid
  %data.ld = load <4 x i8>, ptr addrspace(1) %data.ptr, align 4
  %ele0 = extractelement <4 x i8> %data.ld, i32 0
  %ele1 = extractelement <4 x i8> %data.ld, i32 1
  %ele2 = extractelement <4 x i8> %data.ld, i32 2
  %ele3 = extractelement <4 x i8> %data.ld, i32 3
  %zext0 = zext i8 %ele0 to i32
  %zext1 = zext i8 %ele1 to i32
  %zext2 = zext i8 %ele2 to i32
  %zext3 = zext i8 %ele3 to i32
  %sum1 = add i32 %zext0, %zext1
  %sum2 = xor i32 %sum1, %zext2
  %sum3 = and i32 %sum2, %zext3
  %output.ptr = getelementptr inbounds i32, ptr addrspace(1) %output, i64 %gid
  store i32 %sum3, ptr addrspace(1) %output.ptr, align 4
  ret void
}

; CHECK: void @__vecz_v4_squashv2i32(
; CHECK:  %[[DATA:.+]] = load <2 x i32>
; CHECK:  %[[FREEZE:.+]] = freeze <2 x i32> %[[DATA]]
; CHECK:  %[[SQUASH:.+]] = bitcast <2 x i32> %[[FREEZE]] to i64
; CHECK:  %[[ZEXT0:.+]] = and i64 %[[SQUASH]], 4294967295
; CHECK:  %[[EXTR1:.+]] = lshr i64 %[[SQUASH]], 32
; CHECK:  %[[ZEXT1:.+]] = and i64 %[[EXTR1]], 4294967295
; CHECK:  %[[SUM1:.+]] = add i64 %[[ZEXT0]], %[[ZEXT1]]
define spir_kernel void @squashv2i32(ptr addrspace(1) %data, ptr addrspace(1) %output) #0 {
entry:
  %gid = call i64 @__mux_get_global_id(i64 0) #1
  %data.ptr = getelementptr inbounds <2 x i32>, ptr addrspace(1) %data, i64 %gid
  %data.ld = load <2 x i32>, ptr addrspace(1) %data.ptr, align 4
  %ele0 = extractelement <2 x i32> %data.ld, i32 0
  %ele1 = extractelement <2 x i32> %data.ld, i32 1
  %zext0 = zext i32 %ele0 to i64
  %zext1 = zext i32 %ele1 to i64
  %sum = add i64 %zext0, %zext1
  %output.ptr = getelementptr inbounds i64, ptr addrspace(1) %output, i64 %gid
  store i64 %sum, ptr addrspace(1) %output.ptr, align 4
  ret void
}

; Check we don't squash vectors we consider too large.
; CHECK: void @__vecz_v4_squashv8i32(
; CHECK-NOT: bitcast
define spir_kernel void @squashv8i32(ptr addrspace(1) %data, ptr addrspace(1) %output) #0 {
entry:
  %gid = call i64 @__mux_get_global_id(i64 0) #1
  %data.ptr = getelementptr inbounds <8 x i32>, ptr addrspace(1) %data, i64 %gid
  %data.ld = load <8 x i32>, ptr addrspace(1) %data.ptr, align 32
  %ele0 = extractelement <8 x i32> %data.ld, i32 0
  %ele1 = extractelement <8 x i32> %data.ld, i32 1
  %zext0 = zext i32 %ele0 to i256
  %zext1 = zext i32 %ele1 to i256
  %sum = add i256 %zext0, %zext1
  %output.ptr = getelementptr inbounds i256, ptr addrspace(1) %output, i64 %gid
  store i256 %sum, ptr addrspace(1) %output.ptr, align 32
  ret void
}

; Check we don't squash vectors we consider too large.
; CHECK: void @__vecz_v4_squashv4i64(
; CHECK-NOT: bitcast
define spir_kernel void @squashv4i64(ptr addrspace(1) %data, ptr addrspace(1) %output) #0 {
entry:
  %gid = call i64 @__mux_get_global_id(i64 0) #1
  %data.ptr = getelementptr inbounds <4 x i64>, ptr addrspace(1) %data, i64 %gid
  %data.ld = load <4 x i64>, ptr addrspace(1) %data.ptr, align 32
  %ele0 = extractelement <4 x i64> %data.ld, i32 0
  %ele1 = extractelement <4 x i64> %data.ld, i32 1
  %zext0 = zext i64 %ele0 to i256
  %zext1 = zext i64 %ele1 to i256
  %sum = add i256 %zext0, %zext1
  %output.ptr = getelementptr inbounds i256, ptr addrspace(1) %output, i64 %gid
  store i256 %sum, ptr addrspace(1) %output.ptr, align 32
  ret void
}

declare i64 @__mux_get_global_id(i64)

attributes #0 = { nounwind }
attributes #1 = { nobuiltin nounwind }
