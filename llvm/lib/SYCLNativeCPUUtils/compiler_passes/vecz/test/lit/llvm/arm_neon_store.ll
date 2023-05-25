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

; REQUIRES: arm

; RUN: veczc -k short3_char3_codegen -vecz-simd-width=4 -S < %s | FileCheck %s

; ModuleID = 'Unknown buffer'
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7-unknown-linux-gnueabihf"

; Function Attrs: nounwind
define spir_kernel void @short3_char3_codegen(i8 addrspace(1)* %src, i16 addrspace(1)* %dest) #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %call = call spir_func i32 @_Z13get_global_idj(i32 0) #3
  %call1 = call spir_func <3 x i8> @_Z6vload3jPU3AS1Kc(i32 %call, i8 addrspace(1)* %src) #3
  %call3 = call spir_func <3 x i16> @_Z14convert_short3Dv3_c(<3 x i8> %call1) #3
  call spir_func void @_Z7vstore3Dv3_sjPU3AS1s(<3 x i16> %call3, i32 %call, i16 addrspace(1)* %dest) #3
  ret void
}

declare spir_func i32 @_Z13get_global_idj(i32) #1

declare spir_func <3 x i8> @_Z6vload3jPU3AS1Kc(i32, i8 addrspace(1)*) #1

declare spir_func <3 x i16> @_Z14convert_short3Dv3_c(<3 x i8>) #1

declare spir_func void @_Z7vstore3Dv3_sjPU3AS1s(<3 x i16>, i32, i16 addrspace(1)*) #1

; Function Attrs: inlinehint nounwind
declare spir_func signext i16 @_Z13convert_shortc(i8 signext) #2

; Function Attrs: inlinehint nounwind
declare spir_func <16 x i16> @_Z15convert_short16Dv16_c(<16 x i8>) #2

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { inlinehint nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nobuiltin nounwind }

!opencl.spir.version = !{!0, !0, !0, !0, !0}
!opencl.ocl.version = !{!1, !1, !1, !1, !1}

!0 = !{i32 2, i32 0}
!1 = !{i32 1, i32 2}
!2 = !{i32 1, i32 1}
!3 = !{!"none", !"none"}
!4 = !{!"char*", !"short*"}
!5 = !{!"", !""}

; Assert call to neon intrinsic exists
; CHECK: call void @llvm.arm.neon.vst3.p1.v4i16
