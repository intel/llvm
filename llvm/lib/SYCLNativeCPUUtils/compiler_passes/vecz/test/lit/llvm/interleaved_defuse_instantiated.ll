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

; RUN: veczc -w 4 -vecz-passes=cfg-convert,packetizer -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; CHECK: @.str = private unnamed_addr addrspace(2) constant [8 x i8] c"blah %d\00", align 1
@.str = private unnamed_addr addrspace(2) constant [8 x i8] c"blah %d\00", align 1
@.strf = private unnamed_addr addrspace(2) constant [7 x i8] c"%#16A\0A\00", align 1

; Function Attrs: nounwind
; CHECK-LABEL: define spir_kernel void @__vecz_v4_printf_kernel(
; CHECK: if.then:
; CHECK: [[ELT0:%.*]] = extractelement
; CHECK: [[ELT1:%.*]] = extractelement
; CHECK: [[ELT2:%.*]] = extractelement
; CHECK: [[ELT3:%.*]] = extractelement
; CHECK: = call spir_func i32 @__vecz_b_masked_printf_u3ptrU3AS2jb(ptr addrspace(2) @.str, i32 [[ELT0]]
; CHECK: = call spir_func i32 @__vecz_b_masked_printf_u3ptrU3AS2jb(ptr addrspace(2) @.str, i32 [[ELT1]]
; CHECK: = call spir_func i32 @__vecz_b_masked_printf_u3ptrU3AS2jb(ptr addrspace(2) @.str, i32 [[ELT2]]
; CHECK: = call spir_func i32 @__vecz_b_masked_printf_u3ptrU3AS2jb(ptr addrspace(2) @.str, i32 [[ELT3]]
; CHECK: ret void
define spir_kernel void @printf_kernel(i32 addrspace(1)* %in, i32 addrspace(1)* %stridesX, i32 addrspace(1)* %dst, i32 %width, i32 %height) #0 {
entry:
  %call = call i64 @__mux_get_global_id(i32 0) #3
  %cmp = icmp eq i64 %call, 13
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %call
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %call1 = call spir_func i32 (i8 addrspace(2)*, ...) @printf(i8 addrspace(2)* getelementptr inbounds ([8 x i8], [8
 x i8] addrspace(2)* @.str, i64 0, i64 0), i32 %0) #3
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

declare i64 @__mux_get_global_id(i32) #1

declare extern_weak spir_func i32 @printf(i8 addrspace(2)*, ...) #1

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nobuiltin nounwind }

!opencl.kernels = !{!0}
!llvm.ident = !{!6}

!0 = !{void (i32 addrspace(1)*, i32 addrspace(1)*, i32 addrspace(1)*, i32, i32)* @printf_kernel, !1, !2, !3, !4, !5}
!1 = !{!"kernel_arg_addr_space", i32 1, i32 1, i32 1, i32 0, i32 0}
!2 = !{!"kernel_arg_access_qual", !"none", !"none", !"none", !"none", !"none"}
!3 = !{!"kernel_arg_type", !"int*", !"int*", !"int*", !"int", !"int"}
!4 = !{!"kernel_arg_base_type", !"int*", !"int*", !"int*", !"int", !"int"}
!5 = !{!"kernel_arg_type_qual", !"", !"", !"", !"", !""}
!6 = !{!"clang version 3.8.0 "}
