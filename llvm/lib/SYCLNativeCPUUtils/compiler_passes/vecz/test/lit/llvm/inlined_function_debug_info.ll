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

; Check VECZ debug info for inlined DILocation metadata nodes

; RUN: %pp-llvm-ver -o %t < %s --llvm-ver %LLVMVER
; RUN: veczc -k functions_one -vecz-passes=builtin-inlining -vecz-simd-width=4 -S < %s | FileCheck %t

; ModuleID = '/tmp/inlined_function.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"

; Function Attrs: alwaysinline
define spir_func i32 @k_one(i32 %x, i32 %y) #0 !dbg !4 {
entry:
  call void @llvm.dbg.value(metadata i32 %x, i64 0, metadata !9, metadata !38), !dbg !39
  call void @llvm.dbg.value(metadata i32 %y, i64 0, metadata !10, metadata !38), !dbg !39
  %mul = mul nsw i32 %x, %y, !dbg !40
  ret i32 %mul, !dbg !40
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
define spir_kernel void @functions_one(i32 addrspace(1)* %in1i, i32 addrspace(1)* %in2i, float addrspace(1)* %in1f, float addrspace(1)* %in2f, i32 addrspace(1)* %out1i, float addrspace(1)* %out1f) #2 !dbg !11 {
entry:
  call void @llvm.dbg.value(metadata i32 addrspace(1)* %in1i, i64 0, metadata !18, metadata !38), !dbg !41
  call void @llvm.dbg.value(metadata i32 addrspace(1)* %in2i, i64 0, metadata !19, metadata !38), !dbg !41
  call void @llvm.dbg.value(metadata float addrspace(1)* %in1f, i64 0, metadata !20, metadata !38), !dbg !41
  call void @llvm.dbg.value(metadata float addrspace(1)* %in2f, i64 0, metadata !21, metadata !38), !dbg !41
  call void @llvm.dbg.value(metadata i32 addrspace(1)* %out1i, i64 0, metadata !22, metadata !38), !dbg !41
  call void @llvm.dbg.value(metadata float addrspace(1)* %out1f, i64 0, metadata !23, metadata !38), !dbg !41
  %call = call i64 @__mux_get_global_id(i32 0) #4, !dbg !42
  call void @llvm.dbg.value(metadata i64 %call, i64 0, metadata !24, metadata !38), !dbg !42
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %in1i, i64 %call, !dbg !43
  %0 = load i32, i32 addrspace(1)* %arrayidx, align 4, !dbg !43
  %arrayidx1 = getelementptr inbounds i32, i32 addrspace(1)* %in2i, i64 %call, !dbg !43
  %1 = load i32, i32 addrspace(1)* %arrayidx1, align 4, !dbg !43
  call void @llvm.dbg.value(metadata i32 %0, i64 0, metadata !9, metadata !38), !dbg !44
  call void @llvm.dbg.value(metadata i32 %1, i64 0, metadata !10, metadata !38), !dbg !44
  %mul.i = mul nsw i32 %0, %1, !dbg !46
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %out1i, i64 %call, !dbg !43
  store i32 %mul.i, i32 addrspace(1)* %arrayidx3, align 4, !dbg !43
  ret void, !dbg !47
}

declare i64 @__mux_get_global_id(i32) #3

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { alwaysinline }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nobuiltin }

!llvm.dbg.cu = !{!0}
!opencl.kernels = !{!29}
!llvm.module.flags = !{!36}
!llvm.ident = !{!37}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.1 ", isOptimized: true, runtimeVersion: 0, emissionKind: 1, enums: !2)
!1 = !DIFile(filename: "kernel.opencl", directory: "Aorta/vecz_build")
!2 = !{}
!3 = !{!4, !11}
!4 = distinct !DISubprogram(name: "k_one", scope: !1, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !8)
!5 = !DISubroutineType(types: !6)
!6 = !{!7, !7, !7}
!7 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!8 = !{!9, !10}
!9 = !DILocalVariable(name: "x", arg: 1, scope: !4, file: !1, line: 1, type: !7)
!10 = !DILocalVariable(name: "y", arg: 2, scope: !4, file: !1, line: 1, type: !7)
!11 = distinct !DISubprogram(name: "functions_one", scope: !1, file: !1, line: 6, type: !12, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !17)
!12 = !DISubroutineType(types: !13)
!13 = !{null, !14, !14, !15, !15, !14, !15}
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, align: 64)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 64, align: 64)
!16 = !DIBasicType(name: "float", size: 32, align: 32, encoding: DW_ATE_float)
!17 = !{!18, !19, !20, !21, !22, !23, !24}
!18 = !DILocalVariable(name: "in1i", arg: 1, scope: !11, file: !1, line: 6, type: !14)
!19 = !DILocalVariable(name: "in2i", arg: 2, scope: !11, file: !1, line: 6, type: !14)
!20 = !DILocalVariable(name: "in1f", arg: 3, scope: !11, file: !1, line: 6, type: !15)
!21 = !DILocalVariable(name: "in2f", arg: 4, scope: !11, file: !1, line: 6, type: !15)
!22 = !DILocalVariable(name: "out1i", arg: 5, scope: !11, file: !1, line: 6, type: !14)
!23 = !DILocalVariable(name: "out1f", arg: 6, scope: !11, file: !1, line: 6, type: !15)
!24 = !DILocalVariable(name: "tid", scope: !11, file: !1, line: 7, type: !25)
!25 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !26, line: 33, baseType: !27)
!26 = !DIFile(filename: "Aorta/OCL/modules/builtins/include/builtins/builtins.h", directory: "Aorta/vecz_build")
!27 = !DIDerivedType(tag: DW_TAG_typedef, name: "ulong", file: !26, line: 31, baseType: !28)
!28 = !DIBasicType(name: "long unsigned int", size: 64, align: 64, encoding: DW_ATE_unsigned)
!29 = !{void (i32 addrspace(1)*, i32 addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32 addrspace(1)*, float addrspace(1)*)* @functions_one, !30, !31, !32, !33, !34, !35}
!30 = !{!"kernel_arg_addr_space", i32 1, i32 1, i32 1, i32 1, i32 1, i32 1}
!31 = !{!"kernel_arg_access_qual", !"none", !"none", !"none", !"none", !"none", !"none"}
!32 = !{!"kernel_arg_type", !"int*", !"int*", !"float*", !"float*", !"int*", !"float*"}
!33 = !{!"kernel_arg_base_type", !"int*", !"int*", !"float*", !"float*", !"int*", !"float*"}
!34 = !{!"kernel_arg_type_qual", !"", !"", !"", !"", !"", !""}
!35 = !{!"reqd_work_group_size", i32 32, i32 1, i32 1}
!36 = !{i32 2, !"Debug Info Version", i32 3}
!37 = !{!"clang version 3.8.1 "}
!38 = !DIExpression()
!39 = !DILocation(line: 1, scope: !4)
!40 = !DILocation(line: 2, scope: !4)
!41 = !DILocation(line: 6, scope: !11)
!42 = !DILocation(line: 7, scope: !11)
!43 = !DILocation(line: 8, scope: !11)
!44 = !DILocation(line: 1, scope: !4, inlinedAt: !45)
!45 = distinct !DILocation(line: 8, scope: !11)
!46 = !DILocation(line: 2, scope: !4, inlinedAt: !45)
!47 = !DILocation(line: 9, scope: !11)

; CHECK: spir_func i32 @k_one
; CHECK-SAME: !dbg [[HELPER_DI:![0-9]+]]

; CHECK: define spir_kernel void @__vecz_v4_functions_one
; CHECK-SAME: !dbg [[KERN_DI:![0-9]+]]

; CHECK: %[[LOAD1:[0-9]+]] = load i32, ptr addrspace(1) %{{.*}}, align 4
; CHECK: %[[LOAD2:[0-9]+]] = load i32, ptr addrspace(1) %{{.*}}, align 4
; CHECK-GE19: #dbg_value(i32 %[[LOAD1]], !{{[0-9]+}}, !DIExpression(), [[DI_LOC1:![0-9]+]]
; CHECK-LT19: call void @llvm.dbg.value(metadata i32 %[[LOAD1]], metadata !{{[0-9]+}}, metadata !DIExpression()), !dbg [[DI_LOC1:![0-9]+]]
; CHECK-GE19: #dbg_value(i32 %[[LOAD2]], !{{[0-9]+}}, !DIExpression(), [[DI_LOC1]]
; CHECK-LT19: call void @llvm.dbg.value(metadata i32 %[[LOAD2]], metadata !{{[0-9]+}}, metadata !DIExpression()), !dbg [[DI_LOC1]]
; CHECK: %{{.*}} = mul nsw i32 %[[LOAD1]], %[[LOAD2]], !dbg [[DI_LOC2:![0-9]+]]

; CHECK: [[HELPER_SUBPROGRAM:![0-9]+]] = distinct !DISubprogram(name: "k_one",

; CHECK: [[DI_LOC1]] = !DILocation(line: 1, scope: [[HELPER_SUBPROGRAM]], inlinedAt: [[DI_INLINED_AT:![0-9]+]])
; CHECK: [[DI_INLINED_AT]] = distinct !DILocation(line: 8,
; CHECK: [[DI_LOC2]] = !DILocation(line: 2, scope: [[HELPER_SUBPROGRAM]], inlinedAt: [[DI_INLINED_AT]])
