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

; Regression test for debug info bug related to creating llvm.dbg.value
; intrinsics across all lanes even when scalarization masks disable some
; of the lanes. This occurs when we scalarize insertelement instructions.

; RUN: veczc -k unaligned_load -vecz-passes="function(instcombine,adce),scalarize,packetizer,instcombine" -vecz-simd-width=4 -vecz-choices=FullScalarization -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; CHECK: define spir_kernel void @__vecz_v4_unaligned_load
define spir_kernel void @unaligned_load(i32 addrspace(1)* %in, i32 addrspace(1)* %offsets, i32 addrspace(1)* %out) #0 !dbg !7 {
entry:
  %in.addr = alloca i32 addrspace(1)*, align 8
  %offsets.addr = alloca i32 addrspace(1)*, align 8
  %out.addr = alloca i32 addrspace(1)*, align 8
; CHECK: %tmp = alloca <16 x i32>, align 16
  %tid = alloca i32, align 4
  %tmp = alloca <3 x i32>, align 16
  store i32 addrspace(1)* %in, i32 addrspace(1)** %in.addr, align 8
  call void @llvm.dbg.declare(metadata i32 addrspace(1)** %in.addr, metadata !11, metadata !29), !dbg !30
  store i32 addrspace(1)* %offsets, i32 addrspace(1)** %offsets.addr, align 8
  call void @llvm.dbg.declare(metadata i32 addrspace(1)** %offsets.addr, metadata !12, metadata !29), !dbg !30
  store i32 addrspace(1)* %out, i32 addrspace(1)** %out.addr, align 8
  call void @llvm.dbg.declare(metadata i32 addrspace(1)** %out.addr, metadata !13, metadata !29), !dbg !30
  call void @llvm.dbg.declare(metadata i32* %tid, metadata !14, metadata !29), !dbg !31
  %call = call i64 @__mux_get_global_id(i32 0) #3, !dbg !31
  %conv = trunc i64 %call to i32, !dbg !31
  store i32 %conv, i32* %tid, align 4, !dbg !31
  call void @llvm.dbg.declare(metadata <3 x i32>* %tmp, metadata !15, metadata !29), !dbg !32
  %0 = load i32 addrspace(1)*, i32 addrspace(1)** %in.addr, align 8, !dbg !32
; CHECK: %[[TMP_LD:.+]] = call <4 x i32> @__vecz_b_interleaved_load4_4_Dv4_ju3ptr(ptr nonnull %tmp)
; FIXME: This llvm.dbg.value marks a 'kill location' and denotes the
; termination of the previous value assigned to %tmp - we could probably do
; better here by manifesting a vectorized value?
; CHECK: call void @llvm.dbg.value(metadata i32 {{(poison|undef)}}, metadata [[VAR:![0-9]+]],
; CHECK-SAME:   metadata !DIExpression({{.*}})), !dbg !{{[0-9]+}}
  %1 = load i32, i32* %tid, align 4, !dbg !32
  %mul = mul nsw i32 3, %1, !dbg !32
  %idx.ext = sext i32 %mul to i64, !dbg !32
  %add.ptr = getelementptr inbounds i32, i32 addrspace(1)* %0, i64 %idx.ext, !dbg !32
  %call1 = call spir_func <3 x i32> @_Z6vload3mPKU3AS1i(i64 0, i32 addrspace(1)* %add.ptr) #3, !dbg !32
  %extractVec = shufflevector <3 x i32> %call1, <3 x i32> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 undef>, !dbg !32
  %storetmp = bitcast <3 x i32>* %tmp to <4 x i32>*, !dbg !32
  store <4 x i32> %extractVec, <4 x i32>* %storetmp, align 16, !dbg !32
  %2 = load <3 x i32>, <3 x i32>* %tmp, align 16, !dbg !33
  %3 = extractelement <3 x i32> %2, i64 0, !dbg !33
  %4 = load i32, i32* %tid, align 4, !dbg !33
  %mul2 = mul nsw i32 3, %4, !dbg !33
  %idxprom = sext i32 %mul2 to i64, !dbg !33
  %5 = load i32 addrspace(1)*, i32 addrspace(1)** %out.addr, align 8, !dbg !33
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %5, i64 %idxprom, !dbg !33
  store i32 %3, i32 addrspace(1)* %arrayidx, align 4, !dbg !33
  %6 = load <3 x i32>, <3 x i32>* %tmp, align 16, !dbg !34
  %7 = extractelement <3 x i32> %6, i64 1, !dbg !34
  %8 = load i32, i32* %tid, align 4, !dbg !34
  %mul3 = mul nsw i32 3, %8, !dbg !34
  %add = add nsw i32 %mul3, 1, !dbg !34
  %idxprom4 = sext i32 %add to i64, !dbg !34
  %9 = load i32 addrspace(1)*, i32 addrspace(1)** %out.addr, align 8, !dbg !34
  %arrayidx5 = getelementptr inbounds i32, i32 addrspace(1)* %9, i64 %idxprom4, !dbg !34
  store i32 %7, i32 addrspace(1)* %arrayidx5, align 4, !dbg !34
  %10 = load <3 x i32>, <3 x i32>* %tmp, align 16, !dbg !35
  %11 = extractelement <3 x i32> %10, i64 2, !dbg !35
  %12 = load i32, i32* %tid, align 4, !dbg !35
  %mul6 = mul nsw i32 3, %12, !dbg !35
  %add7 = add nsw i32 %mul6, 2, !dbg !35
  %idxprom8 = sext i32 %add7 to i64, !dbg !35
  %13 = load i32 addrspace(1)*, i32 addrspace(1)** %out.addr, align 8, !dbg !35
  %arrayidx9 = getelementptr inbounds i32, i32 addrspace(1)* %13, i64 %idxprom8, !dbg !35
  store i32 %11, i32 addrspace(1)* %arrayidx9, align 4, !dbg !35
  ret void, !dbg !36
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i64 @__mux_get_global_id(i32) #2

declare spir_func <3 x i32> @_Z6vload3mPKU3AS1i(i64, i32 addrspace(1)*) #2

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nobuiltin }

!llvm.dbg.cu = !{!0}
!opencl.kernels = !{!21}
!llvm.module.flags = !{!27}
!llvm.ident = !{!28}

; Now check we're actually looking at the right variable.
; CHECK: [[VAR]] = !DILocalVariable(name: "tmp",

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.1 ", isOptimized: true, runtimeVersion: 0, emissionKind: 1, enums: !2, retainedTypes: !3)
!1 = !DIFile(filename: "kernel.opencl", directory: "/home/Aorta/vecz_build")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64, align: 64)
!5 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !{!7}
!7 = distinct !DISubprogram(name: "unaligned_load", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !10)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !4, !4, !4}
!10 = !{!11, !12, !13, !14, !15}
!11 = !DILocalVariable(name: "in", arg: 1, scope: !7, file: !1, line: 1, type: !4)
!12 = !DILocalVariable(name: "offsets", arg: 2, scope: !7, file: !1, line: 1, type: !4)
!13 = !DILocalVariable(name: "out", arg: 3, scope: !7, file: !1, line: 1, type: !4)
!14 = !DILocalVariable(name: "tid", scope: !7, file: !1, line: 2, type: !5)
!15 = !DILocalVariable(name: "tmp", scope: !7, file: !1, line: 3, type: !16)
!16 = !DIDerivedType(tag: DW_TAG_typedef, name: "int3", file: !17, line: 64, baseType: !18)
!17 = !DIFile(filename: "/home//Aorta/OCL/modules/builtins/include/builtins/builtins.h", directory: "/home/Aorta/vecz_build")
!18 = !DICompositeType(tag: DW_TAG_array_type, baseType: !5, size: 128, align: 128, flags: DIFlagVector, elements: !19)
!19 = !{!20}
!20 = !DISubrange(count: 3)
!21 = !{void (i32 addrspace(1)*, i32 addrspace(1)*, i32 addrspace(1)*)* @unaligned_load, !22, !23, !24, !25, !26}
!22 = !{!"kernel_arg_addr_space", i32 1, i32 1, i32 1}
!23 = !{!"kernel_arg_access_qual", !"none", !"none", !"none"}
!24 = !{!"kernel_arg_type", !"int*", !"int*", !"int*"}
!25 = !{!"kernel_arg_base_type", !"int*", !"int*", !"int*"}
!26 = !{!"kernel_arg_type_qual", !"", !"", !""}
!27 = !{i32 2, !"Debug Info Version", i32 3}
!28 = !{!"clang version 3.8.1 "}
!29 = !DIExpression()
!30 = !DILocation(line: 1, scope: !7)
!31 = !DILocation(line: 2, scope: !7)
!32 = !DILocation(line: 3, scope: !7)
!33 = !DILocation(line: 4, scope: !7)
!34 = !DILocation(line: 5, scope: !7)
!35 = !DILocation(line: 6, scope: !7)
!36 = !DILocation(line: 7, scope: !7)
