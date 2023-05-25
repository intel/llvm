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

; RUN: veczc -S < %s | FileCheck %s

target triple = "spir64-unknown-unknown"
target datalayout = "e-p:64:64:64-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.testStruct = type { <3 x i32> }

define spir_kernel void @alloca_alias(i32 addrspace(1)* %out, i32 %index) {
entry:
  %myStructs = alloca [2 x %struct.testStruct], align 16
  %call = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %0 = bitcast [2 x %struct.testStruct]* %myStructs to i8*
  call void @llvm.lifetime.start.p0i8(i64 32, i8* nonnull %0)
  %1 = trunc i64 %call to i32
  %conv = add nuw nsw i32 %1, 2
  %2 = insertelement <4 x i32> undef, i32 %conv, i64 0
  %conv2 = add nuw nsw i32 %1, 3
  %3 = insertelement <4 x i32> %2, i32 %conv2, i64 1
  %4 = insertelement <4 x i32> %3, i32 %1, i64 2
  %i = getelementptr inbounds [2 x %struct.testStruct], [2 x %struct.testStruct]* %myStructs, i64 0, i64 1, i32 0
  %storetmp8 = bitcast <3 x i32>* %i to <4 x i32>*
  store <4 x i32> %4, <4 x i32>* %storetmp8, align 16
  %idxprom = sext i32 %index to i64
  %i9 = getelementptr inbounds [2 x %struct.testStruct], [2 x %struct.testStruct]* %myStructs, i64 0, i64 %idxprom, i32 0
  %castToVec410 = bitcast <3 x i32>* %i9 to <4 x i32>*
  %loadVec411 = load <4 x i32>, <4 x i32>* %castToVec410, align 16
  %extractVec12 = shufflevector <4 x i32> %loadVec411, <4 x i32> poison, <3 x i32> <i32 0, i32 1, i32 2>
  %5 = mul i64 %call, 3
  %vstore_base = getelementptr i32, i32 addrspace(1)* %out, i64 %5
  %vstore_extract = extractelement <3 x i32> %extractVec12, i32 0
  %6 = getelementptr i32, i32 addrspace(1)* %vstore_base, i32 0
  store i32 %vstore_extract, i32 addrspace(1)* %6, align 4
  %vstore_extract1 = extractelement <3 x i32> %extractVec12, i32 1
  %7 = getelementptr i32, i32 addrspace(1)* %vstore_base, i32 1
  store i32 %vstore_extract1, i32 addrspace(1)* %7, align 4
  %vstore_extract2 = extractelement <3 x i32> %extractVec12, i32 2
  %8 = getelementptr i32, i32 addrspace(1)* %vstore_base, i32 2
  store i32 %vstore_extract2, i32 addrspace(1)* %8, align 4
  call void @llvm.lifetime.end.p0i8(i64 32, i8* nonnull %0)
  ret void
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8*)

declare spir_func i64 @_Z13get_global_idj(i32)

declare spir_func void @_Z7vstore3Dv3_imPU3AS1i(<3 x i32>, i64, i32 addrspace(1)*)

declare void @llvm.lifetime.end.p0i8(i64 immarg, i8*)

; CHECK: spir_kernel void @__vecz_v4_alloca_alias
; CHECK: alloca [4 x [2 x %struct.testStruct{{.*}}]]
; CHECK-NOT: = alloca .*
