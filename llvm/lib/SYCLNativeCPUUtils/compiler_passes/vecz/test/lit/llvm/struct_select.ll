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

; RUN: veczc -k test -vecz-simd-width=4 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

%struct_type = type { i32, i32 }

define spir_kernel void @test(%struct_type* %in1, %struct_type* %in2, %struct_type* %out) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %in1p = getelementptr inbounds %struct_type, %struct_type* %in1, i64 %call
  %in2p = getelementptr inbounds %struct_type, %struct_type* %in2, i64 %call
  %outp = getelementptr inbounds %struct_type, %struct_type* %out, i64 %call
  %in1v = load %struct_type, %struct_type* %in1p
  %in2v = load %struct_type, %struct_type* %in2p
  %mod = urem i64 %call, 3
  %cmp = icmp eq i64 %mod, 0
  %res = select i1 %cmp, %struct_type %in1v, %struct_type %in2v
  store %struct_type %res, %struct_type* %outp
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)
declare void @llvm.memset.p0i8.i32(i8*,i8,i32,i32,i1)

; CHECK: define spir_kernel void @__vecz_v4_test

; CHECK: select i1 %{{.+}}, %struct_type %{{.+}}, %struct_type %{{.+}}
; CHECK: select i1 %{{.+}}, %struct_type %{{.+}}, %struct_type %{{.+}}
; CHECK: select i1 %{{.+}}, %struct_type %{{.+}}, %struct_type %{{.+}}
; CHECK: select i1 %{{.+}}, %struct_type %{{.+}}, %struct_type %{{.+}}

; CHECK: ret void
