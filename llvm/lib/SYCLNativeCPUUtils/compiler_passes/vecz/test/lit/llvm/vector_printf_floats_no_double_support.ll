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

; RUN: veczc -k test_float_vectors -vecz-simd-width=4 -vecz-double-support=false -vecz-choices=FullScalarization -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

@.str = private unnamed_addr addrspace(2) constant [10 x i8] c"%#4v4hho\0A\00", align 1
@.str32 = private unnamed_addr addrspace(2) constant [11 x i8] c"%#4v32hho\0A\00", align 1
@.str64 = private unnamed_addr addrspace(2) constant [11 x i8] c"%#4v64hho\0A\00", align 1
@.strfv = private unnamed_addr addrspace(2) constant [11 x i8] c"%#16v2hlA\0A\00", align 1

; Function Attrs: nounwind
define spir_kernel void @test(<4 x i8>* %out, <4 x i8>* %in1, <4 x i8>* %in2) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds <4 x i8>, <4 x i8>* %in1, i64 %call
  %0 = load <4 x i8>, <4 x i8>* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds <4 x i8>, <4 x i8>* %in2, i64 %call
  %1 = load <4 x i8>, <4 x i8>* %arrayidx1, align 4
  %add = add <4 x i8> %1, %0
  %arrayidx2 = getelementptr inbounds <4 x i8>, <4 x i8>* %out, i64 %call
  store <4 x i8> %add, <4 x i8>* %arrayidx2, align 4
  %call4 = call spir_func i32 (i8 addrspace(2)*, ...) @printf(i8 addrspace(2)* getelementptr inbounds ([10 x i8], [10 x i8] addrspace(2)* @.str, i64 0, i64 0), <4 x i8> %add)
  ret void
}

define spir_kernel void @test32(<32 x i8>* %out, <32 x i8>* %in1, <32 x i8>* %in2) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds <32 x i8>, <32 x i8>* %in1, i64 %call
  %0 = load <32 x i8>, <32 x i8>* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds <32 x i8>, <32 x i8>* %in2, i64 %call
  %1 = load <32 x i8>, <32 x i8>* %arrayidx1, align 4
  %add = add <32 x i8> %1, %0
  %arrayidx2 = getelementptr inbounds <32 x i8>, <32 x i8>* %out, i64 %call
  store <32 x i8> %add, <32 x i8>* %arrayidx2, align 4
  %call4 = call spir_func i32 (i8 addrspace(2)*, ...) @printf(i8 addrspace(2)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(2)* @.str32, i64 0, i64 0), <32 x i8> %add)
  ret void
}

define spir_kernel void @test64(<64 x i8>* %out, <64 x i8>* %in1, <64 x i8>* %in2) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds <64 x i8>, <64 x i8>* %in1, i64 %call
  %0 = load <64 x i8>, <64 x i8>* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds <64 x i8>, <64 x i8>* %in2, i64 %call
  %1 = load <64 x i8>, <64 x i8>* %arrayidx1, align 4
  %add = add <64 x i8> %1, %0
  %arrayidx2 = getelementptr inbounds <64 x i8>, <64 x i8>* %out, i64 %call
  store <64 x i8> %add, <64 x i8>* %arrayidx2, align 4
  %call4 = call spir_func i32 (i8 addrspace(2)*, ...) @printf(i8 addrspace(2)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(2)* @.str64, i64 0, i64 0), <64 x i8> %add)
  ret void
}

define spir_kernel void @test_float_vectors(<2 x float>* %in) {
entry:
  %call = call spir_func i64 @_Z13get_global_idj(i32 0)
  %arrayidx = getelementptr inbounds <2 x float>, <2 x float>* %in, i64 %call
  %0 = load <2 x float>, <2 x float>* %arrayidx, align 8
  %mul = fmul <2 x float> %0, %0
  %call8 = call spir_func i32 (i8 addrspace(2)*, ...) @printf(i8 addrspace(2)* getelementptr inbounds ([11 x i8], [11 x i8] addrspace(2)* @.strfv, i64 0, i64 0), <2 x float> %mul)
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)

declare extern_weak spir_func i32 @printf(i8 addrspace(2)*, ...)

; CHECK: @[[STR:.+]] = private unnamed_addr addrspace(2) constant [13 x i8] c"%#16A,%#16A\0A\00", align 1

; CHECK: define spir_kernel void @__vecz_v4_test_float_vectors
; CHECK: %[[V5:[0-9]+]] = extractelement <4 x float> %{{.+}}, {{(i32|i64)}} 0
; CHECK: %[[V6:[0-9]+]] = extractelement <4 x float> %{{.+}}, {{(i32|i64)}} 1
; CHECK: %[[V7:[0-9]+]] = extractelement <4 x float> %{{.+}}, {{(i32|i64)}} 2
; CHECK: %[[V8:[0-9]+]] = extractelement <4 x float> %{{.+}}, {{(i32|i64)}} 3
; CHECK: %[[V10:[0-9]+]] = extractelement <4 x float> %{{.+}}, {{(i32|i64)}} 0
; CHECK: %[[V11:[0-9]+]] = extractelement <4 x float> %{{.+}}, {{(i32|i64)}} 1
; CHECK: %[[V12:[0-9]+]] = extractelement <4 x float> %{{.+}}, {{(i32|i64)}} 2
; CHECK: %[[V13:[0-9]+]] = extractelement <4 x float> %{{.+}}, {{(i32|i64)}} 3
; CHECK: call spir_func i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) @[[STR]], float %[[V5]], float %[[V10]])
; CHECK: call spir_func i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) @[[STR]], float %[[V6]], float %[[V11]])
; CHECK: call spir_func i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) @[[STR]], float %[[V7]], float %[[V12]])
; CHECK: call spir_func i32 (ptr addrspace(2), ...) @printf(ptr addrspace(2) @[[STR]], float %[[V8]], float %[[V13]])
; CHECK: ret void
