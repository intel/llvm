; Copyright (C) Codeplay Software Limited
;
; Licensed under the Apache License, Version 2.0 (the "License") with LLVM
; Exceptions; you may not use this file except in compliance with the License.
; You may obtain a copy of the License at
;
;     https://github.com/uxlfoundation/oneapi-construction-kit/blob/main/LICENSE.txt
;
; Unless required by applicable law or agreed to in writing, software
; distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
; WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
; License for the specific language governing permissions and limitations
; under the License.
;
; SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

; RUN: veczc -k test -vecz-simd-width=4 -S < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

%struct_type = type { i32, i32 }

define spir_kernel void @test(i32* %in, i32* %out, %struct_type* %sin) {
entry:
  %call = call i64 @__mux_get_global_id(i32 0)
  %inp = getelementptr inbounds i32, i32* %in, i64 %call
  %oup = getelementptr inbounds i32, i32* %out, i64 %call
  %o = load i32, i32* %oup
  ; do this little compare + phi to throw off the InstCombine pass and ensure
  ; we end up with a phi %struct_type that must be instantiated
  %s = insertvalue %struct_type poison, i32 %o, 1
  %cmpcall = icmp ult i64 16, %call
  br i1 %cmpcall, label %lower, label %higher

lower:
  %lowers = insertvalue %struct_type %s, i32 0, 0
  br label %lower.higher.phi

higher:
  %highers = insertvalue %struct_type %s, i32 1, 0
  br label %lower.higher.phi

lower.higher.phi:
  %lowerhigherstruct = phi %struct_type [%lowers, %lower], [%highers, %higher]
  br label %for.cond

for.cond:
  %storemerge = phi %struct_type [ %incv, %for.inc ], [ %lowerhigherstruct, %lower.higher.phi ]
  %s1 = extractvalue %struct_type %storemerge, 1
  %s1ext = zext i32 %s1 to i64
  %cmp = icmp ult i64 %s1ext, %call
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %l = load i32, i32* %inp, align 4
  store i32 %l, i32* %oup, align 4
  br label %for.inc

for.inc:
  %toadd = extractvalue %struct_type %storemerge, 1
  %toadd64 = zext i32 %toadd to i64
  %ca = add i64 %toadd64, %call
  %sinp = getelementptr inbounds %struct_type, %struct_type* %sin, i64 %ca
  %sinv = load %struct_type, %struct_type* %sinp
  %sinintv = extractvalue %struct_type %sinv, 1
  %incv = insertvalue %struct_type %storemerge, i32 %sinintv, 1
  br label %for.cond

for.end:
  ret void
}

declare i64 @__mux_get_global_id(i32)
declare void @llvm.memset.p0i8.i32(i8*,i8,i32,i32,i1)

; CHECK: define spir_kernel void @__vecz_v4_test

; Check if the struct creation has been instantiated
; CHECK: %[[V2:[0-9]+]] = load <4 x i32>, ptr %oup, align 4
; CHECK: %[[V3:[0-9]+]] = extractelement <4 x i32> %[[V2]], {{(i32|i64)}} 0
; CHECK: %[[V4:[0-9]+]] = extractelement <4 x i32> %[[V2]], {{(i32|i64)}} 1
; CHECK: %[[V5:[0-9]+]] = extractelement <4 x i32> %[[V2]], {{(i32|i64)}} 2
; CHECK: %[[V6:[0-9]+]] = extractelement <4 x i32> %[[V2]], {{(i32|i64)}} 3
; CHECK: %[[S24:.+]] = insertvalue %struct_type poison, i32 %[[V3]], 1
; CHECK: %[[S25:.+]] = insertvalue %struct_type poison, i32 %[[V4]], 1
; CHECK: %[[S26:.+]] = insertvalue %struct_type poison, i32 %[[V5]], 1
; CHECK: %[[S27:.+]] = insertvalue %struct_type poison, i32 %[[V6]], 1

; Check if the phi node has been instantiated
; CHECK: phi %struct_type [ %{{.+}}, %entry ], [ %{{.+}}, %for.cond ]
; CHECK: phi %struct_type [ %{{.+}}, %entry ], [ %{{.+}}, %for.cond ]
; CHECK: phi %struct_type [ %{{.+}}, %entry ], [ %{{.+}}, %for.cond ]
; CHECK: phi %struct_type [ %{{.+}}, %entry ], [ %{{.+}}, %for.cond ]
; CHECK: extractvalue %struct_type %{{.+}}, 1
; CHECK: extractvalue %struct_type %{{.+}}, 1
; CHECK: extractvalue %struct_type %{{.+}}, 1
; CHECK: extractvalue %struct_type %{{.+}}, 1

; Check if the operations that use integer types are vectorized
; CHECK: zext <4 x i32>
; CHECK: icmp ugt <4 x i64>
; CHECK: select <4 x i1>
; CHECK: %[[L423:.+]] = call <4 x i32> @__vecz_b_masked_load4_Dv4_ju3ptrDv4_b(ptr %{{.*}}, <4 x i1>
; CHECK: call void @__vecz_b_masked_store4_Dv4_ju3ptrDv4_b(<4 x i32> %[[L423]], ptr{{( nonnull)? %.*}}, <4 x i1>

; CHECK: ret void
