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

; RUN: veczc -k foo3 -vecz-simd-width=4 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define void @foo1(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %call = call i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %0, i32 addrspace(1)* %arrayidx1, align 4
  ret void
}

declare i64 @__mux_get_global_id(i32)

define void @foo2(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  call void @foo1(i32 addrspace(1)* %in, i32 addrspace(1)* %out)
  ret void
}

define spir_kernel void @foo3(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  call void @foo2(i32 addrspace(1)* %in, i32 addrspace(1)* %out)
  ret void
}

; CHECK: define spir_kernel void @__vecz_v4_foo3(ptr addrspace(1) %in, ptr addrspace(1) %out)
; CHECK-NOT: call spir_kernel
; CHECK: call i64 @__mux_get_global_id(i32 0)
; CHECK: load <4 x i32>, ptr addrspace(1) %{{.+}}, align 4
; CHECK: store <4 x i32> %{{.+}}, ptr addrspace(1) %{{.+}}, align 4
; CHECK: ret void
