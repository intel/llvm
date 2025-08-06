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

; Check that debug info intrinsics aren't created using undef values.
; These cause the backend to assert in codegen.

; RUN: veczc -k test_fn -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
define spir_kernel void @test_fn(i16 addrspace(1)* %src, <4 x i16> addrspace(1)* %dst) #0 !dbg !4 {
entry:
  %src.addr = alloca i16 addrspace(1)*, align 8
  %dst.addr = alloca <4 x i16> addrspace(1)*, align 8
  %tid = alloca i32, align 4
  %tmp = alloca <4 x i16>, align 8
  store i16 addrspace(1)* %src, i16 addrspace(1)** %src.addr, align 8
  call void @llvm.dbg.declare(metadata i16 addrspace(1)** %src.addr, metadata !18, metadata !32), !dbg !33
  store <4 x i16> addrspace(1)* %dst, <4 x i16> addrspace(1)** %dst.addr, align 8
  call void @llvm.dbg.declare(metadata <4 x i16> addrspace(1)** %dst.addr, metadata !19, metadata !32), !dbg !33
  call void @llvm.dbg.declare(metadata i32* %tid, metadata !20, metadata !32), !dbg !34
  %call = call i64 @__mux_get_global_id(i32 0) #3, !dbg !34
  %conv = trunc i64 %call to i32, !dbg !34
  store i32 %conv, i32* %tid, align 4, !dbg !34
  call void @llvm.dbg.declare(metadata <4 x i16>* %tmp, metadata !22, metadata !32), !dbg !35
  %0 = load i32, i32* %tid, align 4, !dbg !35
  %conv1 = sext i32 %0 to i64, !dbg !35
  %1 = load i16 addrspace(1)*, i16 addrspace(1)** %src.addr, align 8, !dbg !35
  %call2 = call spir_func <3 x i16> @_Z6vload3mPKU3AS1t(i64 %conv1, i16 addrspace(1)* %1) #3, !dbg !35
  %call3 = call spir_func <4 x i16> @_Z9as_short4Dv3_t(<3 x i16> %call2) #3, !dbg !35
  store <4 x i16> %call3, <4 x i16>* %tmp, align 8, !dbg !35
  %2 = load <4 x i16>, <4 x i16>* %tmp, align 8, !dbg !36
  %3 = load i32, i32* %tid, align 4, !dbg !36
  %idxprom = sext i32 %3 to i64, !dbg !36
  %4 = load <4 x i16> addrspace(1)*, <4 x i16> addrspace(1)** %dst.addr, align 8, !dbg !36
  %arrayidx = getelementptr inbounds <4 x i16>, <4 x i16> addrspace(1)* %4, i64 %idxprom, !dbg !36
  store <4 x i16> %2, <4 x i16> addrspace(1)* %arrayidx, align 8, !dbg !36
  ret void, !dbg !37
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i64 @__mux_get_global_id(i32) #2

declare spir_func <4 x i16> @_Z9as_short4Dv3_t(<3 x i16>) #2

declare spir_func <3 x i16> @_Z6vload3mPKU3AS1t(i64, i16 addrspace(1)*) #2

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nobuiltin }

!llvm.dbg.cu = !{!0}
!opencl.kernels = !{!23}
!llvm.module.flags = !{!30}
!llvm.ident = !{!31}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.1 ", isOptimized: true, runtimeVersion: 0, emissionKind: 1, enums: !2)
!1 = !DIFile(filename: "kernel.opencl", directory: "/tmp")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "test_fn", scope: !1, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !17)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7, !11}
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64, align: 64)
!8 = !DIDerivedType(tag: DW_TAG_typedef, name: "ushort", file: !9, line: 29, baseType: !10)
!9 = !DIFile(filename: "builtins/include/builtins/builtins.h", directory: "/tmp")
!10 = !DIBasicType(name: "unsigned short", size: 16, align: 16, encoding: DW_ATE_unsigned)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64, align: 64)
!12 = !DIDerivedType(tag: DW_TAG_typedef, name: "short4", file: !9, line: 55, baseType: !13)
!13 = !DICompositeType(tag: DW_TAG_array_type, baseType: !14, size: 64, align: 64, flags: DIFlagVector, elements: !15)
!14 = !DIBasicType(name: "short", size: 16, align: 16, encoding: DW_ATE_signed)
!15 = !{!16}
!16 = !DISubrange(count: 4)
!17 = !{!18, !19, !20, !22}
!18 = !DILocalVariable(name: "src", arg: 1, scope: !4, file: !1, line: 2, type: !7)
!19 = !DILocalVariable(name: "dst", arg: 2, scope: !4, file: !1, line: 2, type: !11)
!20 = !DILocalVariable(name: "tid", scope: !4, file: !1, line: 4, type: !21)
!21 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!22 = !DILocalVariable(name: "tmp", scope: !4, file: !1, line: 5, type: !12)
!23 = !{void (i16 addrspace(1)*, <4 x i16> addrspace(1)*)* @test_fn, !24, !25, !26, !27, !28, !29}
!24 = !{!"kernel_arg_addr_space", i32 1, i32 1}
!25 = !{!"kernel_arg_access_qual", !"none", !"none"}
!26 = !{!"kernel_arg_type", !"ushort*", !"short4*"}
!27 = !{!"kernel_arg_base_type", !"ushort*", !"short __attribute__((ext_vector_type(4)))*"}
!28 = !{!"kernel_arg_type_qual", !"", !""}
!29 = !{!"reqd_work_group_size", i32 32, i32 1, i32 1}
!30 = !{i32 2, !"Debug Info Version", i32 3}
!31 = !{!"clang version 3.8.1 "}
!32 = !DIExpression()
!33 = !DILocation(line: 2, scope: !4)
!34 = !DILocation(line: 4, scope: !4)
!35 = !DILocation(line: 5, scope: !4)
!36 = !DILocation(line: 6, scope: !4)
!37 = !DILocation(line: 7, scope: !4)

; Vectorized kernel function
; CHECK: @__vecz_v[[WIDTH:[0-9]+]]_test_fn({{.*}} !dbg {{![0-9]+}}
