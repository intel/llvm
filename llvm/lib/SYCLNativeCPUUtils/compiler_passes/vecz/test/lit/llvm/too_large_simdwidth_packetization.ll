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

; RUN: veczc -vecz-simd-width=128 -S < %s | FileCheck %s

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
; CHECK-LABEL: define spir_kernel void @__vecz_v128_add(ptr addrspace(1) %in1, ptr addrspace(1) %in2, ptr addrspace(1) %out) 
; CHECK: = load <128 x i32>, ptr addrspace(1)
; CHECK: = load <128 x i32>, ptr addrspace(1)
; CHECK: = add nsw <128 x i32>
; CHECK: store <128 x i32>
define spir_kernel void @add(i32 addrspace(1)* %in1, i32 addrspace(1)* %in2, i32 addrspace(1)* %out) #0 !dbg !4 {
entry:
  %in1.addr = alloca i32 addrspace(1)*, align 8
  %in2.addr = alloca i32 addrspace(1)*, align 8
  %out.addr = alloca i32 addrspace(1)*, align 8
  %tid = alloca i64, align 8
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  store i32 addrspace(1)* %in1, i32 addrspace(1)** %in1.addr, align 8
  call void @llvm.dbg.declare(metadata i32 addrspace(1)** %in1.addr, metadata !11, metadata !29), !dbg !30
  store i32 addrspace(1)* %in2, i32 addrspace(1)** %in2.addr, align 8
  call void @llvm.dbg.declare(metadata i32 addrspace(1)** %in2.addr, metadata !12, metadata !29), !dbg !30
  store i32 addrspace(1)* %out, i32 addrspace(1)** %out.addr, align 8
  call void @llvm.dbg.declare(metadata i32 addrspace(1)** %out.addr, metadata !13, metadata !29), !dbg !30
  call void @llvm.dbg.declare(metadata i64* %tid, metadata !14, metadata !29), !dbg !31
  %call = call i64 @__mux_get_global_id(i32 0) #3, !dbg !31
  store i64 %call, i64* %tid, align 8, !dbg !31
  call void @llvm.dbg.declare(metadata i32* %a, metadata !19, metadata !29), !dbg !32
  %0 = load i64, i64* %tid, align 8, !dbg !32
  %1 = load i32 addrspace(1)*, i32 addrspace(1)** %in1.addr, align 8, !dbg !32
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %1, i64 %0, !dbg !32
  %2 = load i32, i32 addrspace(1)* %arrayidx, align 4, !dbg !32
  store i32 %2, i32* %a, align 4, !dbg !32
  call void @llvm.dbg.declare(metadata i32* %b, metadata !20, metadata !29), !dbg !33
  %3 = load i64, i64* %tid, align 8, !dbg !33
  %4 = load i32 addrspace(1)*, i32 addrspace(1)** %in2.addr, align 8, !dbg !33
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %4, i64 %3, !dbg !33
  %5 = load i32, i32 addrspace(1)* %arrayidx1, align 4, !dbg !33
  store i32 %5, i32* %b, align 4, !dbg !33
  %6 = load i32, i32* %a, align 4, !dbg !34
  %7 = load i32, i32* %b, align 4, !dbg !34
  %add = add nsw i32 %6, %7, !dbg !34
  %8 = load i64, i64* %tid, align 8, !dbg !34
  %9 = load i32 addrspace(1)*, i32 addrspace(1)** %out.addr, align 8, !dbg !34
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %9, i64 %8, !dbg !34
  store i32 %add, i32 addrspace(1)* %arrayidx2, align 4, !dbg !34
  ret void, !dbg !35
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i64 @__mux_get_global_id(i32) #2

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nobuiltin }

!llvm.dbg.cu = !{!0}
!opencl.kernels = !{!21}
!llvm.module.flags = !{!27}
!llvm.ident = !{!28}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: 1, enums: !2)
!1 = !DIFile(filename: "<stdin>", directory: "/tmp")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "add", scope: !5, file: !5, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !10)
!5 = !DIFile(filename: "kernel.opencl", directory: "/tmp")
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8, !8, !8}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64, align: 64)
!9 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !{!11, !12, !13, !14, !19, !20}
!11 = !DILocalVariable(name: "in1", arg: 1, scope: !4, file: !5, line: 1, type: !8)
!12 = !DILocalVariable(name: "in2", arg: 2, scope: !4, file: !5, line: 1, type: !8)
!13 = !DILocalVariable(name: "out", arg: 3, scope: !4, file: !5, line: 1, type: !8)
!14 = !DILocalVariable(name: "tid", scope: !4, file: !5, line: 3, type: !15)
!15 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !16, line: 33, baseType: !17)
!16 = !DIFile(filename: "/Aorta/OCL/modules/builtins/include/builtins/builtins.h", directory: "/tmp")
!17 = !DIDerivedType(tag: DW_TAG_typedef, name: "ulong", file: !16, line: 31, baseType: !18)
!18 = !DIBasicType(name: "long unsigned int", size: 64, align: 64, encoding: DW_ATE_unsigned)
!19 = !DILocalVariable(name: "a", scope: !4, file: !5, line: 5, type: !9)
!20 = !DILocalVariable(name: "b", scope: !4, file: !5, line: 6, type: !9)
!21 = !{void (i32 addrspace(1)*, i32 addrspace(1)*, i32 addrspace(1)*)* @add, !22, !23, !24, !25, !26}
!22 = !{!"kernel_arg_addr_space", i32 1, i32 1, i32 1}
!23 = !{!"kernel_arg_access_qual", !"none", !"none", !"none"}
!24 = !{!"kernel_arg_type", !"int*", !"int*", !"int*"}
!25 = !{!"kernel_arg_base_type", !"int*", !"int*", !"int*"}
!26 = !{!"kernel_arg_type_qual", !"", !"", !""}
!27 = !{i32 2, !"Debug Info Version", i32 3}
!28 = !{!"clang version 3.8.0 "}
!29 = !DIExpression()
!30 = !DILocation(line: 1, scope: !4)
!31 = !DILocation(line: 3, scope: !4)
!32 = !DILocation(line: 5, scope: !4)
!33 = !DILocation(line: 6, scope: !4)
!34 = !DILocation(line: 7, scope: !4)
!35 = !DILocation(line: 8, scope: !4)
