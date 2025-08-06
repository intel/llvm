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

; RUN: veczc -k squash -vecz-choices=TargetIndependentPacketization -vecz-passes="squash-small-vecs,function(dce),packetizer" -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "E-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @squash(<4 x i8> addrspace(1)* %data, i32 addrspace(1)* %output) #0 {
entry:
  %gid = call i64 @__mux_get_global_id(i64 0) #2
  %data.ptr = getelementptr inbounds <4 x i8>, <4 x i8> addrspace(1)* %data, i64 %gid
  %data.ld = load <4 x i8>, <4 x i8> addrspace(1)* %data.ptr, align 8
  %ele0 = extractelement <4 x i8> %data.ld, i32 3
  %ele1 = extractelement <4 x i8> %data.ld, i32 2
  %ele2 = extractelement <4 x i8> %data.ld, i32 1
  %ele3 = extractelement <4 x i8> %data.ld, i32 0
  %zext0 = sext i8 %ele0 to i32
  %zext1 = sext i8 %ele1 to i32
  %zext2 = sext i8 %ele2 to i32
  %zext3 = sext i8 %ele3 to i32
  %sum1 = add i32 %zext0, %zext1
  %sum2 = xor i32 %sum1, %zext2
  %sum3 = and i32 %sum2, %zext3
  %output.ptr = getelementptr inbounds i32, i32 addrspace(1)* %output, i64 %gid
  store i32 %sum3, i32 addrspace(1)* %output.ptr, align 8
  ret void
}

declare i64 @__mux_get_global_id(i64) #1

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nobuiltin nounwind }

; It checks that the <4 x i8> is converted into a i32 and uses shifts
; to implement the extract elements and sexts.
;
; CHECK: void @__vecz_v4_squash
; CHECK:  %[[DATA:.+]] = load <16 x i8>
; CHECK-NOT: shufflevector
; CHECK:  %[[FREEZE:.+]] = freeze <16 x i8> %[[DATA]]
; CHECK:  %[[SQUASH:.+]] = bitcast <16 x i8> %[[FREEZE]] to <4 x i32>
; CHECK:  %[[EXTR0:.+]] = shl <4 x i32> %[[SQUASH]], {{<(i32 24(, )?)+>|splat \(i32 24\)}}
; CHECK:  %[[SEXT0:.+]] = ashr <4 x i32> %[[EXTR0]], {{<(i32 24(, )?)+>|splat \(i32 24\)}}
; CHECK:  %[[EXTR1:.+]] = shl <4 x i32> %[[SQUASH]], {{<(i32 16(, )?)+>|splat \(i32 16\)}}
; CHECK:  %[[SEXT1:.+]] = ashr <4 x i32> %[[EXTR1]], {{<(i32 24(, )?)+>|splat \(i32 24\)}}
; CHECK:  %[[EXTR2:.+]] = shl <4 x i32> %[[SQUASH]], {{<(i32 8(, )?)+>|splat \(i32 8\)}}
; CHECK:  %[[SEXT2:.+]] = ashr <4 x i32> %[[EXTR2]], {{<(i32 24(, )?)+>|splat \(i32 24\)}}
; CHECK:  %[[SEXT3:.+]] = ashr <4 x i32> %[[SQUASH]], {{<(i32 24(, )?)+>|splat \(i32 24\)}}
; CHECK:  %[[SUM1:.+]] = add <4 x i32> %[[SEXT0]], %[[SEXT1]]
; CHECK:  %[[SUM2:.+]] = xor <4 x i32> %[[SUM1]], %[[SEXT2]]
; CHECK:  %[[SUM3:.+]] = and <4 x i32> %[[SUM2]], %[[SEXT3]]
; CHECK:  ret void
