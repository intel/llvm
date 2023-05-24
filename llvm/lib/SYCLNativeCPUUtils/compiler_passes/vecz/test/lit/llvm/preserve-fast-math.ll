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

; RUN: %veczc -S -vecz-passes=packetizer < %s | %filecheck %s

; CHECK: %{{.*}} = fcmp nnan ninf olt <4 x float> %{{.*}}, %{{.*}}

define spir_kernel void @fast_nan(float addrspace(1)* %src1, float addrspace(1)* %src2, i16 addrspace(1)* %dst, i32 %width) {
entry:
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds float, float addrspace(1)* %src1, i64 %call
  %0 = load float, float addrspace(1)* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds float, float addrspace(1)* %src2, i64 %call
  %1 = load float, float addrspace(1)* %arrayidx2, align 4
  %cmp = fcmp nnan ninf olt float %0, %1
  %conv4 = zext i1 %cmp to i16
  %arrayidx6 = getelementptr inbounds i16, i16 addrspace(1)* %dst, i64 %call
  store i16 %conv4, i16 addrspace(1)* %arrayidx6, align 2
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)
