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

; REQUIRES: llvm-12+
; RUN: veczc -k foo -w 2 -vecz-passes scalarize,mask-memops,packetizer -print-after mask-memops -S < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare i64 @__mux_get_global_id(i32)

; CHECK: IR Dump After Simplify masked memory operations{{( on __vecz_v2_foo)?}}
; CHECK-NEXT: define spir_kernel void @__vecz_v2_foo(ptr addrspace(1) %out) #0 {
; CHECK-NEXT:   %idx = call i64 @__mux_get_global_id(i32 0)
; CHECK-NEXT:   %arrayidx = getelementptr i32, ptr addrspace(1) %out, i64 %idx
; CHECK-NEXT:   store i32 0, ptr addrspace(1) %arrayidx, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define spir_kernel void @__vecz_v2_foo(ptr addrspace(1) %out) {{.*}} {
; CHECK-NEXT:   %idx = call i64 @__mux_get_global_id(i32 0)
; CHECK-NEXT:   %arrayidx = getelementptr i32, ptr addrspace(1) %out, i64 %idx
; CHECK-NEXT:   store <2 x i32> zeroinitializer, ptr addrspace(1) %arrayidx, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

define spir_kernel void @foo(i32 addrspace(1)* %out) {
  %idx = call i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %idx
  store i32 0, i32 addrspace(1)* %arrayidx, align 4
  ret void
}
