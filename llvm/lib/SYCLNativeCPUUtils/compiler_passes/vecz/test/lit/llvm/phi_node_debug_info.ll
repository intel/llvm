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

; Check that debug info intrinsics are correctly placed after
; phi nodes.

; RUN: %pp-llvm-ver -o %t < %s --llvm-ver %LLVMVER
; RUN: veczc -vecz-simd-width=4 -S < %s | FileCheck %t

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
; CHECK: define spir_kernel void @__vecz_v4_loop_phi(
define spir_kernel void @loop_phi(i32 addrspace(3)* %a, i32 addrspace(3)* %b) #0 !dbg !4 {
entry:
  %a.addr = alloca i32 addrspace(3)*, align 8
  %b.addr = alloca i32 addrspace(3)*, align 8
  %tid = alloca i64, align 8
  %i = alloca i32, align 4
  store i32 addrspace(3)* %a, i32 addrspace(3)** %a.addr, align 8
  call void @llvm.dbg.declare(metadata i32 addrspace(3)** %a.addr, metadata !12, metadata !30), !dbg !31
  store i32 addrspace(3)* %b, i32 addrspace(3)** %b.addr, align 8
  call void @llvm.dbg.declare(metadata i32 addrspace(3)** %b.addr, metadata !13, metadata !30), !dbg !31
  call void @llvm.dbg.declare(metadata i64* %tid, metadata !14, metadata !30), !dbg !32
  %call = call i64 @__mux_get_local_id(i32 0) #3, !dbg !32
  store i64 %call, i64* %tid, align 8, !dbg !32
  call void @llvm.dbg.declare(metadata i32* %i, metadata !19, metadata !30), !dbg !33
  %0 = load i64, i64* %tid, align 8, !dbg !33
  %conv = trunc i64 %0 to i32, !dbg !33
  store i32 %conv, i32* %i, align 4, !dbg !33
  br label %for.cond, !dbg !33


; CHECK: for.cond:
; CHECK: %[[PHI1:.+]] = phi {{i[0-9]+}} [ %{{.+}}, %entry ], [ %{{.+}}, %for.cond ]
; CHECK-GE19: #dbg_value(i64 %[[PHI1]], !{{[0-9]+}},
; CHECK-LT19: call void @llvm.dbg.value(metadata i64 %[[PHI1]], metadata !{{[0-9]+}},
; CHECK-SAME: !DIExpression({{.*}}),
; CHECK-SAME: !{{[0-9]+}}
; Check we haven't inserted a llvm.dbg.value intrinsic before the last of the PHIs.
; CHECK-NOT: phi
for.cond:                                         ; preds = %for.inc, %entry
  %1 = load i32, i32* %i, align 4, !dbg !34
  %cmp = icmp slt i32 %1, 128, !dbg !34
  br i1 %cmp, label %for.body, label %for.end, !dbg !33

for.body:                                         ; preds = %for.cond
  %2 = load i32, i32* %i, align 4, !dbg !36
  %idxprom = sext i32 %2 to i64, !dbg !36
  %3 = load i32 addrspace(3)*, i32 addrspace(3)** %b.addr, align 8, !dbg !36
  %arrayidx = getelementptr inbounds i32, i32 addrspace(3)* %3, i64 %idxprom, !dbg !36
  %4 = load i32, i32 addrspace(3)* %arrayidx, align 4, !dbg !36
  %5 = load i32, i32* %i, align 4, !dbg !36
  %idxprom2 = sext i32 %5 to i64, !dbg !36
  %6 = load i32 addrspace(3)*, i32 addrspace(3)** %a.addr, align 8, !dbg !36
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(3)* %6, i64 %idxprom2, !dbg !36
  store i32 %4, i32 addrspace(3)* %arrayidx3, align 4, !dbg !36
  br label %for.inc, !dbg !38

for.inc:                                          ; preds = %for.body
  %7 = load i32, i32* %i, align 4, !dbg !34
  %add = add nsw i32 %7, 32, !dbg !34
  store i32 %add, i32* %i, align 4, !dbg !34
  br label %for.cond, !dbg !34

for.end:                                          ; preds = %for.cond
; CHECK: ret void
  ret void, !dbg !39
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i64 @__mux_get_local_id(i32) #2

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nobuiltin }

!llvm.dbg.cu = !{!0}
!opencl.kernels = !{!21}
!llvm.module.flags = !{!28}
!llvm.ident = !{!29}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.1 ", isOptimized: true, runtimeVersion: 0, emissionKind: 1, enums: !2)
!1 = !DIFile(filename: "kernel.opencl", directory: "/home/Aorta/build")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "loop_phi", scope: !1, file: !1, line: 2, type: !5, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !11)
!5 = !DISubroutineType(types: !6)
!6 = !{null, !7, !9}
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64, align: 64)
!8 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 64, align: 64)
!10 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !8)
!11 = !{!12, !13, !14, !19}
!12 = !DILocalVariable(name: "a", arg: 1, scope: !4, file: !1, line: 2, type: !7)
!13 = !DILocalVariable(name: "b", arg: 2, scope: !4, file: !1, line: 2, type: !9)
!14 = !DILocalVariable(name: "tid", scope: !4, file: !1, line: 3, type: !15)
!15 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !16, line: 33, baseType: !17)
!16 = !DIFile(filename: "/home/Aorta/OCL/modules/builtins/include/builtins/builtins.h", directory: "/home/Aorta/build")
!17 = !DIDerivedType(tag: DW_TAG_typedef, name: "ulong", file: !16, line: 31, baseType: !18)
!18 = !DIBasicType(name: "long unsigned int", size: 64, align: 64, encoding: DW_ATE_unsigned)
!19 = !DILocalVariable(name: "i", scope: !20, file: !1, line: 4, type: !8)
!20 = distinct !DILexicalBlock(scope: !4, file: !1, line: 4)
!21 = !{void (i32 addrspace(3)*, i32 addrspace(3)*)* @loop_phi, !22, !23, !24, !25, !26, !27}
!22 = !{!"kernel_arg_addr_space", i32 3, i32 3}
!23 = !{!"kernel_arg_access_qual", !"none", !"none"}
!24 = !{!"kernel_arg_type", !"int*", !"int*"}
!25 = !{!"kernel_arg_base_type", !"int*", !"int*"}
!26 = !{!"kernel_arg_type_qual", !"", !"const"}
!27 = !{!"reqd_work_group_size", i32 32, i32 1, i32 1}
!28 = !{i32 2, !"Debug Info Version", i32 3}
!29 = !{!"clang version 3.8.1 "}
!30 = !DIExpression()
!31 = !DILocation(line: 2, scope: !4)
!32 = !DILocation(line: 3, scope: !4)
!33 = !DILocation(line: 4, scope: !20)
!34 = !DILocation(line: 4, scope: !35)
!35 = distinct !DILexicalBlock(scope: !20, file: !1, line: 4)
!36 = !DILocation(line: 5, scope: !37)
!37 = distinct !DILexicalBlock(scope: !35, file: !1, line: 4)
!38 = !DILocation(line: 6, scope: !37)
!39 = !DILocation(line: 7, scope: !4)
