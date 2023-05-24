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

; RUN: %veczc -k multiple_dimensions_0 -vecz-simd-width 4 -S < %s | %filecheck %s

; ModuleID = 'kernel.opencl'
source_filename = "kernel.opencl"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind readonly
declare spir_func i64 @_Z13get_global_idj(i32) #1

; Function Attrs: convergent nounwind readonly
declare spir_func i64 @_Z15get_global_sizej(i32) #1

; Function Attrs: convergent nounwind
define spir_kernel void @multiple_dimensions_0(i32 addrspace(1)* %output) #2 {
entry:
  %call.i = call spir_func i64 @_Z13get_global_idj(i32 0) #3
  %call1.i = call spir_func i64 @_Z15get_global_sizej(i32 1) #3
  %mul.i = mul i64 %call1.i, %call.i
  %call2.i = call spir_func i64 @_Z15get_global_sizej(i32 2) #3
  %mul3.i = mul i64 %mul.i, %call2.i
  %call4.i = call spir_func i64 @_Z13get_global_idj(i32 1) #3
  %mul6.i = mul i64 %call2.i, %call4.i
  %add.i = add i64 %mul6.i, %mul3.i
  %call7.i = call spir_func i64 @_Z13get_global_idj(i32 2) #3
  %add8.i = add i64 %add.i, %call7.i
  %conv = trunc i64 %add8.i to i32
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %output, i64 %add8.i
  store i32 %conv, i32 addrspace(1)* %arrayidx, align 4
  ret void
}

attributes #0 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { convergent nounwind readonly "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { convergent nobuiltin nounwind readonly }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}
!opencl.kernels = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{!"clang version 8.0.0 (https://github.com/llvm-mirror/clang.git bfbe338a893dde6ba65b2bed6ffea1652a592819) (https://github.com/llvm-mirror/llvm.git a99d6d2122ca2f208e1c4bcaf02ff5930f244f34)"}
!3 = !{void (i32 addrspace(1)*)* @multiple_dimensions_0, !4, !5, !6, !7, !8, !9}
!4 = !{!"kernel_arg_addr_space", i32 1}
!5 = !{!"kernel_arg_access_qual", !"none"}
!6 = !{!"kernel_arg_type", !"int*"}
!7 = !{!"kernel_arg_base_type", !"int*"}
!8 = !{!"kernel_arg_type_qual", !""}
!9 = !{!"kernel_arg_name", !"output"}

; Function start
; CHECK: define spir_kernel void @__vecz_v4_multiple_dimensions_0

; make sure the stride calculation uses the correct operand of the multiply
; CHECK: %[[CALL1:.+]] = call spir_func i64 @_Z15get_global_sizej(i32 1)
; CHECK: %[[CALL2:.+]] = call spir_func i64 @_Z15get_global_sizej(i32 2)
; CHECK: %[[NEWMUL:.+]] = mul i64 %[[CALL1]], %[[CALL2]]
; CHECK: call void @__vecz_b_interleaved_store4_V_Dv4_ju3ptrU3AS1({{.+}} %[[NEWMUL]])
