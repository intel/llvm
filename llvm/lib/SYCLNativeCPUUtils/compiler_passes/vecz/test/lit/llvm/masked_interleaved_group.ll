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

; RUN: veczc -k mask -vecz-simd-width=16 -vecz-choices=TargetIndependentPacketization -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
source_filename = "kernel.opencl"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_kernel void @mask(i8 addrspace(1)* %out, i8 addrspace(1)* %in) #0 {
entry:
  %call = call i64 @__mux_get_global_id(i32 0) #2
  %call.tr = trunc i64 %call to i32
  %conv = shl i32 %call.tr, 1
  %idx.ext = sext i32 %conv to i64
  %add.ptr = getelementptr inbounds i8, i8 addrspace(1)* %in, i64 %idx.ext
  %0 = load i8, i8 addrspace(1)* %add.ptr, align 1
  %arrayidx1 = getelementptr inbounds i8, i8 addrspace(1)* %add.ptr, i64 1
  %1 = load i8, i8 addrspace(1)* %arrayidx1, align 1
  %add.ptr3 = getelementptr inbounds i8, i8 addrspace(1)* %out, i64 %idx.ext
  %conv4 = sext i8 %0 to i32
  %conv5 = sext i8 %1 to i32
  %add = add nsw i32 %conv5, %conv4
  %cmp = icmp slt i32 %add, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %arrayidx7 = getelementptr inbounds i8, i8 addrspace(1)* %add.ptr3, i64 1
  store i8 %0, i8 addrspace(1)* %arrayidx7, align 1
  br label %if.end

if.else:                                          ; preds = %entry
  store i8 %1, i8 addrspace(1)* %add.ptr3, align 1
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

; Function Attrs: convergent nounwind readonly
declare i64 @__mux_get_global_id(i32) #1

attributes #0 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nobuiltin nounwind readonly }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}
!opencl.kernels = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{!"clang version 8.0.0 (https://github.com/llvm-mirror/clang.git bfbe338a893dde6ba65b2bed6ffea1652a592819) (https://github.com/llvm-mirror/llvm.git a99d6d2122ca2f208e1c4bcaf02ff5930f244f34)"}
!3 = !{void (i8 addrspace(1)*, i8 addrspace(1)*)* @mask, !4, !5, !6, !7, !8, !9}
!4 = !{!"kernel_arg_addr_space", i32 1, i32 1}
!5 = !{!"kernel_arg_access_qual", !"none", !"none"}
!6 = !{!"kernel_arg_type", !"char*", !"char*"}
!7 = !{!"kernel_arg_base_type", !"char*", !"char*"}
!8 = !{!"kernel_arg_type_qual", !"", !""}
!9 = !{!"kernel_arg_name", !"out", !"in"}

; This test makes sure we combine a group of masked interleaved stores
; into a single masked interleaved store using interleave operations.
; We expect the interleaved stores to come out unaltered.

; CHECK: entry:

; The data to store gets interleaved:
; CHECK: %interleave{{.*}} = shufflevector <16 x i8>
; CHECK: %interleave{{.*}} = shufflevector <16 x i8>

; The masks get interleaved:
; CHECK: %interleave{{.*}} = shufflevector <16 x i1>
; CHECK: %interleave{{.*}} = shufflevector <16 x i1>

; The stores are masked stores:
; CHECK: call void @llvm.masked.store.v16i8.p1(<16 x i8>
; CHECK: call void @llvm.masked.store.v16i8.p1(<16 x i8>

; Definitely no unmasked stores:
; CHECK-NOT: store <16 x i8>
; CHECK: ret void
