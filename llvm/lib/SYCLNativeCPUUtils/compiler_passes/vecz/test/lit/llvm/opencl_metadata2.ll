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

; RUN: veczc -k second_test -vecz-simd-width=4 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

define spir_kernel void @test(i32 addrspace(2)* %in, i32 addrspace(1)* %out, i8 addrspace(2)* %text, double %f) {
entry:
  %call = call i64 @__mux_get_global_id(i32 0)
  %arrayidx = getelementptr inbounds i32, i32 addrspace(2)* %in, i64 %call
  %0 = load i32, i32 addrspace(2)* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %call
  store i32 %0, i32 addrspace(1)* %arrayidx1, align 4
  ret void
}

define spir_kernel void @second_test(i32 %a, i32 %b) {
entry:
  %call = call i64 @__mux_get_global_id(i32 0)
  ret void
}

declare i64 @__mux_get_global_id(i32)

!opencl.kernels = !{!0, !6}
!opencl.kernel_wg_size_info = !{!12}
!llvm.ident = !{!13}

!0 = !{void (i32 addrspace(2)*, i32 addrspace(1)*, i8 addrspace(2)*, double)* @test, !1, !2, !3, !4, !5}
!1 = !{!"kernel_arg_addr_space", i32 2, i32 1, i32 2, i32 0}
!2 = !{!"kernel_arg_access_qual", !"none", !"none", !"none", !"none"}
!3 = !{!"kernel_arg_type", !"int*", !"int*", !"char*", !"double"}
!4 = !{!"kernel_arg_base_type", !"int*", !"int*", !"char*", !"double"}
!5 = !{!"kernel_arg_type_qual", !"const", !"", !"const", !""}
!6 = !{void (i32, i32)* @second_test, !7, !8, !9, !10, !11}
!7 = !{!"kernel_arg_addr_space", i32 0, i32 0}
!8 = !{!"kernel_arg_access_qual", !"none", !"none"}
!9 = !{!"kernel_arg_type", !"int", !"int"}
!10 = !{!"kernel_arg_base_type", !"int", !"int"}
!11 = !{!"kernel_arg_type_qual", !"", !""}
!12 = !{void (i32 addrspace(2)*, i32 addrspace(1)*, i8 addrspace(2)*, double)* @test, i32 16, i32 1, i32 1, i1 true}
!13 = !{!"clang version 3.8.1 "}

; Sanity checking
; CHECK: define spir_kernel void @second_test(i32 %a, i32 %b)
; CHECK: define spir_kernel void @__vecz_v4_second_test(i32 %a, i32 %b)

; Check if we have the metadata for the kernels
; CHECK: !opencl.kernels = !{![[MD0:[0-9]+]], ![[MD6:[0-9]+]], ![[MD12:[0-9]+]]}
; CHECK: !opencl.kernel_wg_size_info = !{![[MD13:[0-9]+]]}
; CHECK: !llvm.ident = !{![[MD14:[0-9]+]]}

; Check the actual metadata
; CHECK: ![[MD6]] = !{ptr @second_test, ![[MD7:[0-9]+]], ![[MD8:[0-9]+]], ![[MD9:[0-9]+]], ![[MD10:[0-9]+]], ![[MD11:[0-9]+]]}
; CHECK: ![[MD7]] = !{!"kernel_arg_addr_space", i32 0, i32 0}
; CHECK: ![[MD8]] = !{!"kernel_arg_access_qual", !"none", !"none"}
; CHECK: ![[MD9]] = !{!"kernel_arg_type", !"int", !"int"}
; CHECK: ![[MD10]] = !{!"kernel_arg_base_type", !"int", !"int"}
; CHECK: ![[MD11]] = !{!"kernel_arg_type_qual", !"", !""}
; CHECK: ![[MD12]] = !{ptr @__vecz_v4_second_test, ![[MD7]], ![[MD8]], ![[MD9]], ![[MD10]], ![[MD11]]}
; CHECK: ![[MD13]] = !{ptr @test, i32 16, i32 1, i32 1, i1 true}
