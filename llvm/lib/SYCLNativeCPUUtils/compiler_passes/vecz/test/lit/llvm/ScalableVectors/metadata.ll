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

; RUN: veczc -k test -vecz-scalable -vecz-simd-width=8 -S < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir-unknown-unknown"

declare i64 @__mux_get_global_id(i32)

define spir_kernel void @test(i32 addrspace(1)* %in) {
entry:
  %idx = call i64 @__mux_get_global_id(i32 0)
  %load = load i32, i32 addrspace(1)* %in
  %slot = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %idx
  store i32 %load, i32 addrspace(1)* %slot
  ret void
}

; CHECK: define spir_kernel void @test(ptr addrspace(1) %in) !codeplay_ca_vecz.base !0
; CHECK: define spir_kernel void @__vecz_nxv8_test(ptr addrspace(1) %in) #0 !codeplay_ca_vecz.derived !2

; CHECK: attributes #0 = { "mux-base-fn-name"="__vecz_nxv8_test" }

; CHECK: !0 = !{!1, ptr @__vecz_nxv8_test}

; CHECK: !1 = !{i32 8, i32 1, i32 0, i32 0}
; CHECK: !2 = !{!1, ptr @test}
