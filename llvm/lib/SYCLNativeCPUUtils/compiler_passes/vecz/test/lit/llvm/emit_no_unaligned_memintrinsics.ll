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

; RUN: veczc -k entry -vecz-passes=cfg-convert,packetizer -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Laid out, this struct is 80 bytes
%struct.S2 = type { i16, [7 x i32], i32, <16 x i8>, [4 x i32] }

; Function Attrs: norecurse nounwind
define spir_kernel void @entry(i64 addrspace(1)* %result, %struct.S2* %result2) {
entry:
  %gid = call i64 @__mux_get_local_id(i32 0)
  %sa = alloca %struct.S2, align 16
  %sb = alloca %struct.S2, align 16
  %sa_i8 = bitcast %struct.S2* %sa to i8*
  %sb_i8 = bitcast %struct.S2* %sb to i8*
  %sb_i8as = addrspacecast i8* %sb_i8 to i8 addrspace(1)*
  %rsi = ptrtoint i64 addrspace(1)* %result to i64
  %rsit = trunc i64 %rsi to i8
  call void @llvm.memset.p0i8.i64(i8*  %sa_i8, i8 %rsit, i64 80, i32 4, i1 false)
  call void @llvm.memset.p1i8.i64(i8 addrspace(1)*  %sb_i8as, i8 0, i64 80, i32 4, i1 false)
  %lr = addrspacecast %struct.S2* %result2 to %struct.S2 addrspace(1)*
  %lri = bitcast %struct.S2 addrspace(1)* %lr to i64 addrspace(1)*
  %cond = icmp eq i64 addrspace(1)* %result, %lri
  br i1 %cond, label %middle, label %end

middle:
  call void @llvm.memcpy.p1i8.p0i8.i64(i8 addrspace(1)*  %sb_i8as, i8* %sa_i8, i64 80, i32 4, i1 false)
  br label %end

end:
  %g_343 = getelementptr inbounds %struct.S2, %struct.S2* %sa, i64 0, i32 0
  %g_343_load = load i16, i16* %g_343
  %g_343_zext = zext i16 %g_343_load to i64
  %resp = getelementptr i64, i64 addrspace(1)* %result, i64 %gid
  store i64 %g_343_zext, i64 addrspace(1)* %resp, align 8
  %result2_i8 = bitcast %struct.S2* %result2 to i8*
  call void @llvm.memcpy.p0i8.p1i8.i64(i8* %result2_i8, i8 addrspace(1)* %sb_i8as, i64 80, i32 4, i1 false)
  call void @llvm.memcpy.p0i8.p1i8.i64(i8* %result2_i8, i8 addrspace(1)* %sb_i8as, i64 80, i32 4, i1 false)  
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1)
declare void @llvm.memset.p1i8.i64(i8 addrspace(1)* nocapture, i8, i64, i32, i1)

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p1i8.p0i8.i64(i8 addrspace(1)* nocapture, i8* nocapture readonly, i64, i32, i1)
declare void @llvm.memcpy.p0i8.p1i8.i64(i8* nocapture, i8 addrspace(1)* nocapture readonly, i64, i32, i1)

declare i64 @__mux_get_local_id(i32)

; Sanity checks: Make sure the non-vecz entry function is still in place and
; contains memset and memcpy. This is done in order to prevent future bafflement
; in case some pass optimizes them out.
; CHECK: define spir_kernel void @entry
; CHECK: entry:
; CHECK: call void @llvm.memset
; CHECK: call void @llvm.memset
; CHECK: middle:
; CHECK: call void @llvm.memcpy
; CHECK: end:
; CHECK: call void @llvm.memcpy

; And now for the actual checks

; Check if the kernel was vectorized
; CHECK: define spir_kernel void @__vecz_v{{[0-9]+}}_entry

; Check if the memset and memcpy calls are still there
; CHECK: call void @llvm.memset
; CHECK: call void @llvm.memcpy

; End of function
; CHECK: ret void
