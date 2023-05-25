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

; RUN: veczc -k func_10 -vecz-passes="function(mem2reg),vecz-mem2reg" -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

%struct.S2 = type { i16, [7 x i32], i32, <16 x i8>, [4 x i32] }

; Function Attrs: norecurse nounwind
define spir_kernel void @entry(%struct.S2** %result) #0 {
entry:
  %c_640 = alloca %struct.S2, align 16
  %p_639 = alloca %struct.S2*, align 8
  store %struct.S2* %c_640, %struct.S2** %p_639, align 8
  %0 = load %struct.S2*, %struct.S2** %p_639, align 8
  store %struct.S2* %0, %struct.S2** %result, align 8
  ret void
}

define spir_func void @func_10(%struct.S2* %p_484, i64** %ret) {
entry:
  %l_462 = alloca i64, align 8
  %l_461 = alloca i64*, align 8
  %.cast = ptrtoint %struct.S2* %p_484 to i64
  store i64 %.cast, i64* %l_462, align 8
  store i64* %l_462, i64** %l_461, align 8
  store i64* %l_462, i64** %ret, align 8
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) #1

attributes #0 = { norecurse nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }

!llvm.ident = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!opencl.kernels = !{!1}

!0 = !{!"clang version 3.8.1 "}
!1 = !{void (%struct.S2**)* @entry, !2, !3, !4, !5, !6}
!2 = !{!"kernel_arg_addr_space", i32 1}
!3 = !{!"kernel_arg_access_qual", !"none"}
!4 = !{!"kernel_arg_type", !"ulong*"}
!5 = !{!"kernel_arg_base_type", !"ulong*"}
!6 = !{!"kernel_arg_type_qual", !""}

; Check if the alloca used for its pointer is still here
; CHECK: @__vecz_v4_func_10
; CHECK: %l_462 = alloca i64, align 8

; Check that the other alloca(s) have been promoted
; CHECK-NOT: alloca

; Check if the store using the alloca is still here
; CHECK:  store i64 %.cast, ptr %l_462, align 8
