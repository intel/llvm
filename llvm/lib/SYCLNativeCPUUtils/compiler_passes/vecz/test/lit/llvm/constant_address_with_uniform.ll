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

; RUN: veczc -w 4 -S < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir-unknown-unknown"

declare spir_func i32 @__mux_get_global_id(i32);

define spir_kernel void @test(i32 addrspace(1)* %out, i32 addrspace(1)* addrspace(1)* %out2) {
entry:
  %gid = call i32 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 3
  store i32 %gid, i32 addrspace(1)* %arrayidx, align 4

  %arrayidx2 = getelementptr inbounds i32 addrspace(1)*, i32 addrspace(1)* addrspace(1)* %out2, i32 %gid
  store i32 addrspace(1)* %arrayidx, i32 addrspace(1)* addrspace(1)* %arrayidx2, align 4

  ret void
}

; CHECK: define spir_kernel void @__vecz_v4_test
; CHECK-NEXT: entry:
; CHECK-NEXT: %gid = call i32 @__mux_get_global_id(i32 0)
; CHECK-NEXT: %arrayidx = getelementptr inbounds {{i32|i8}}, ptr addrspace(1) %out, i32 {{3|12}}
; CHECK: store i32 %gid, ptr addrspace(1) %arrayidx, align 4
; CHECK: store <4 x ptr addrspace(1)> %{{.+}}, ptr addrspace(1) %{{.+}}
