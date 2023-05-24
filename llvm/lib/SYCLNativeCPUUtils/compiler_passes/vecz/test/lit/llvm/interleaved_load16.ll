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

; RUN: %veczc -k load16 -vecz-simd-width 4 -S < %s | %filecheck %s

; ModuleID = 'Unknown buffer'
source_filename = "kernel.opencl"
target datalayout = "e-m:e-p:32:32-f64:64-i64:64-v128:64-v64:64-v32:32-v16:16-n8:16:32-S64"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_kernel void @load16(i8 addrspace(1)* %out, i8 addrspace(1)* %in, i32 %stride) #0 !shave_original_kernel !10 {
entry:
  %call = call spir_func i32 @_Z13get_global_idj(i32 0) #2
  %call1 = call spir_func i32 @_Z13get_global_idj(i32 1) #2
  %mul = mul nsw i32 %call1, %stride
  %add = add nsw i32 %mul, %call
  %mul2 = shl nsw i32 %add, 1
  %arrayidx = getelementptr inbounds i8, i8 addrspace(1)* %in, i32 %mul2
  %0 = load i8, i8 addrspace(1)* %arrayidx, align 1
  %mul3 = mul nsw i32 %call1, %stride
  %add4 = add nsw i32 %mul3, %call
  %mul5 = shl nsw i32 %add4, 1
  %add6 = add i32 %mul5, 3
  %arrayidx7 = getelementptr inbounds i8, i8 addrspace(1)* %in, i32 %add6
  %1 = load i8, i8 addrspace(1)* %arrayidx7, align 1
  %add9 = add i8 %1, %0
  %mul11 = mul nsw i32 %call1, %stride
  %add12 = add nsw i32 %mul11, %call
  %arrayidx13 = getelementptr inbounds i8, i8 addrspace(1)* %out, i32 %add12
  store i8 %add9, i8 addrspace(1)* %arrayidx13, align 1
  ret void
}

; Function Attrs: convergent nounwind readonly
declare spir_func i32 @_Z13get_global_idj(i32) #1

attributes #0 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nobuiltin nounwind readonly }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}
!opencl.kernels = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{!"clang version 7.0.0 (tags/RELEASE_700/final) (based on LLVM 7.0.0)"}
!3 = !{void (i8 addrspace(1)*, i8 addrspace(1)*, i32)* @load16, !4, !5, !6, !7, !8, !9}
!4 = !{!"kernel_arg_addr_space", i32 1, i32 1, i32 0}
!5 = !{!"kernel_arg_access_qual", !"none", !"none", !"none"}
!6 = !{!"kernel_arg_type", !"uchar*", !"uchar*", !"int"}
!7 = !{!"kernel_arg_base_type", !"uchar*", !"uchar*", !"int"}
!8 = !{!"kernel_arg_type_qual", !"", !"", !""}
!9 = !{!"kernel_arg_name", !"out", !"in", !"stride"}
!10 = !{!"load16"}

; Function start
; CHECK: define spir_kernel void @__vecz_v4_load16

; There should be exactly 2 interleaved loads in the code
; CHECK: call <4 x i8> @__vecz_b_interleaved_load1_2_Dv4_hu3ptrU3AS1
; CHECK: call <4 x i8> @__vecz_b_interleaved_load1_2_Dv4_hu3ptrU3AS1

; There shouldn't be any more interleaved loads or stores left
; CHECK-NOT: call <4 x i8> @__vecz_b_interleaved_load

; There definitely shouldn't be any gather loads
; CHECK-NOT: call <4 x i8> @__vecz_b_gather_load

; Function end
; CHECK: ret void
