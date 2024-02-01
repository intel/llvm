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

; RUN: veczc -k test -w 4 -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

define spir_kernel void @test(i32 addrspace(1)* %out) #0 {
entry:
  %gid = call i64 @__mux_get_global_id(i32 0) #1
  %conv = trunc i64 %gid to i32
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 3
  store i32 %conv, i32 addrspace(1)* %arrayidx, align 4
  ret void
}

declare i64 @__mux_get_global_id(i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!opencl.kernels = !{!0}
!opencl.spir.version = !{!7}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!8}
!opencl.compiler.options = !{!8}

!0 = !{void (i32 addrspace(1)*)* @test, !1, !2, !3, !4, !5, !6}
!1 = !{!"kernel_arg_addr_space", i32 1}
!2 = !{!"kernel_arg_access_qual", !"none"}
!3 = !{!"kernel_arg_type", !"int*"}
!4 = !{!"kernel_arg_base_type", !"int*"}
!5 = !{!"kernel_arg_type_qual", !""}
!6 = !{!"kernel_arg_name", !"out"}
!7 = !{i32 1, i32 2}
!8 = !{}

; CHECK: define spir_kernel void @__vecz_v4_test
; CHECK-NEXT: entry:
; CHECK-NEXT: %gid = call i64 @__mux_get_global_id(i32 0)
; CHECK-NEXT: %conv = trunc i64 %gid to i32
; CHECK-NEXT: %arrayidx = getelementptr inbounds {{i32|i8}}, ptr addrspace(1) %out, i64 {{3|12}}
; CHECK-NEXT: store i32 %conv, ptr addrspace(1) %arrayidx, align 4
