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

; RUN: veczc -k runtime_index -vecz-simd-width=4 -vecz-choices=FullScalarization -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

declare i64 @__mux_get_global_id(i32)

define spir_kernel void @runtime_index(<4 x i32>* %in, <4 x i32>* %out, i32* %index) {
entry:
  %call = call i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds <4 x i32>, <4 x i32>* %in, i64 %call
  %0 = load <4 x i32>, <4 x i32>* %arrayidx
  %arrayidx1 = getelementptr inbounds <4 x i32>, <4 x i32>* %out, i64 %call
  store <4 x i32> %0, <4 x i32>* %arrayidx1
  %arrayidx2 = getelementptr inbounds i32, i32* %index, i64 %call
  %1 = load i32, i32* %arrayidx2
  %arrayidx3 = getelementptr inbounds <4 x i32>, <4 x i32>* %out, i64 %call
  %vecins = insertelement <4 x i32> %0, i32 42, i32 %1
  store <4 x i32> %vecins, <4 x i32>* %arrayidx3
  ret void
}

; CHECK: define spir_kernel void @__vecz_v4_runtime_index

; Four icmps and selects
; CHECK: icmp eq <4 x i32> %{{.+}}, zeroinitializer
; CHECK: select <4 x i1> %{{.+}}, <4 x i32> {{<(i32 42(, )?)+>|splat \(i32 42\)}}
; CHECK: icmp eq <4 x i32> %{{.+}}, {{<(i32 1(, )?)+>|splat \(i32 1\)}}
; CHECK: select <4 x i1> %{{.+}}, <4 x i32> {{<(i32 42(, )?)+>|splat \(i32 42\)}}
; CHECK: icmp eq <4 x i32> %{{.+}}, {{<(i32 2(, )?)+>|splat \(i32 2\)}}
; CHECK: select <4 x i1> %{{.+}}, <4 x i32> {{<(i32 42(, )?)+>|splat \(i32 42\)}}
; CHECK: icmp eq <4 x i32> %{{.+}}, {{<(i32 3(, )?)+>|splat \(i32 3\)}}
; CHECK: select <4 x i1> %{{.+}}, <4 x i32> {{<(i32 42(, )?)+>|splat \(i32 42\)}}

; Four stores
; CHECK: store <4 x i32>
; CHECK: store <4 x i32>
; CHECK: store <4 x i32>
; CHECK: store <4 x i32>
; CHECK: ret void
