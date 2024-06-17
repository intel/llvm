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

; Check that debug info is preserved in the vectorized kernel.
; Specifically that the scalarization pass doesn't destroy DI
; intrinsics attached to the vector instructions it scalarizes.

; RUN: %pp-llvm-ver -o %t < %s --llvm-ver %LLVMVER
; RUN: veczc -k mul2 -vecz-passes="scalarize,function(mem2reg)" -vecz-choices=FullScalarization -S < %s | FileCheck %t

; ModuleID = 'kernel.opencl'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "spir64-unknown-unknown"


; Function Attrs: nounwind
define spir_kernel void @mul2(<2 x i32> addrspace(1)* %in1, <2 x i32> addrspace(1)* %in2, <2 x i32> addrspace(1)* %out) #0 !dbg !4 {
entry:
  %in1.addr = alloca <2 x i32> addrspace(1)*, align 8
  %in2.addr = alloca <2 x i32> addrspace(1)*, align 8
  %out.addr = alloca <2 x i32> addrspace(1)*, align 8
  %tid = alloca i64, align 8
  %a = alloca <2 x i32>, align 8
  %b = alloca <2 x i32>, align 8
  %tmp = alloca <2 x i32>, align 8
  store <2 x i32> addrspace(1)* %in1, <2 x i32> addrspace(1)** %in1.addr, align 8
  call void @llvm.dbg.declare(metadata <2 x i32> addrspace(1)** %in1.addr, metadata !16, metadata !34), !dbg !35
  store <2 x i32> addrspace(1)* %in2, <2 x i32> addrspace(1)** %in2.addr, align 8
  call void @llvm.dbg.declare(metadata <2 x i32> addrspace(1)** %in2.addr, metadata !17, metadata !34), !dbg !35
  store <2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)** %out.addr, align 8
  call void @llvm.dbg.declare(metadata <2 x i32> addrspace(1)** %out.addr, metadata !18, metadata !34), !dbg !35
  call void @llvm.dbg.declare(metadata i64* %tid, metadata !19, metadata !34), !dbg !36
  %call = call i64 @__mux_get_global_id(i32 0) #3, !dbg !36
  store i64 %call, i64* %tid, align 8, !dbg !36
  call void @llvm.dbg.declare(metadata <2 x i32>* %a, metadata !23, metadata !34), !dbg !37
  %0 = load i64, i64* %tid, align 8, !dbg !37
  %1 = load <2 x i32> addrspace(1)*, <2 x i32> addrspace(1)** %in1.addr, align 8, !dbg !37
  %arrayidx = getelementptr inbounds <2 x i32>, <2 x i32> addrspace(1)* %1, i64 %0, !dbg !37
  %2 = load <2 x i32>, <2 x i32> addrspace(1)* %arrayidx, align 8, !dbg !37
  store <2 x i32> %2, <2 x i32>* %a, align 8, !dbg !37
  call void @llvm.dbg.declare(metadata <2 x i32>* %b, metadata !24, metadata !34), !dbg !38
  %3 = load i64, i64* %tid, align 8, !dbg !38
  %4 = load <2 x i32> addrspace(1)*, <2 x i32> addrspace(1)** %in2.addr, align 8, !dbg !38
  %arrayidx1 = getelementptr inbounds <2 x i32>, <2 x i32> addrspace(1)* %4, i64 %3, !dbg !38
  %5 = load <2 x i32>, <2 x i32> addrspace(1)* %arrayidx1, align 8, !dbg !38
  store <2 x i32> %5, <2 x i32>* %b, align 8, !dbg !38
  call void @llvm.dbg.declare(metadata <2 x i32>* %tmp, metadata !25, metadata !34), !dbg !39
  %6 = load <2 x i32>, <2 x i32>* %a, align 8, !dbg !39
  %7 = load <2 x i32>, <2 x i32>* %b, align 8, !dbg !39
  %mul = mul <2 x i32> %6, %7, !dbg !39
  store <2 x i32> %mul, <2 x i32>* %tmp, align 8, !dbg !39
  %8 = load <2 x i32>, <2 x i32>* %tmp, align 8, !dbg !40
  %9 = load i64, i64* %tid, align 8, !dbg !40
  %10 = load <2 x i32> addrspace(1)*, <2 x i32> addrspace(1)** %out.addr, align 8, !dbg !40
  %arrayidx2 = getelementptr inbounds <2 x i32>, <2 x i32> addrspace(1)* %10, i64 %9, !dbg !40
  store <2 x i32> %8, <2 x i32> addrspace(1)* %arrayidx2, align 8, !dbg !40
  ret void, !dbg !41
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare i64 @__mux_get_global_id(i32) #2

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="0" "stackrealign" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nobuiltin }

!llvm.dbg.cu = !{!0}
!opencl.kernels = !{!26}
!llvm.module.flags = !{!32}
!llvm.ident = !{!33}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: 1, enums: !2)
!1 = !DIFile(filename: "<stdin>", directory: "Aorta/vecz_build")
!2 = !{}
!3 = !{!4}
!4 = distinct !DISubprogram(name: "mul2", scope: !5, file: !5, line: 1, type: !6, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !15)
!5 = !DIFile(filename: "kernel.opencl", directory: "Aorta/vecz_build")
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8, !8, !8}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64, align: 64)
!9 = !DIDerivedType(tag: DW_TAG_typedef, name: "int2", file: !10, line: 63, baseType: !11)
!10 = !DIFile(filename: "Aorta/OCL/modules/builtins/include/builtins/builtins.h", directory: "Aorta/vecz_build")
!11 = !DICompositeType(tag: DW_TAG_array_type, baseType: !12, size: 64, align: 64, flags: DIFlagVector, elements: !13)
!12 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !{!14}
!14 = !DISubrange(count: 2)
!15 = !{!16, !17, !18, !19, !23, !24, !25}
!16 = !DILocalVariable(name: "in1", arg: 1, scope: !4, file: !5, line: 1, type: !8)
!17 = !DILocalVariable(name: "in2", arg: 2, scope: !4, file: !5, line: 1, type: !8)
!18 = !DILocalVariable(name: "out", arg: 3, scope: !4, file: !5, line: 1, type: !8)
!19 = !DILocalVariable(name: "tid", scope: !4, file: !5, line: 3, type: !20)
!20 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !10, line: 33, baseType: !21)
!21 = !DIDerivedType(tag: DW_TAG_typedef, name: "ulong", file: !10, line: 31, baseType: !22)
!22 = !DIBasicType(name: "long unsigned int", size: 64, align: 64, encoding: DW_ATE_unsigned)
!23 = !DILocalVariable(name: "a", scope: !4, file: !5, line: 4, type: !9)
!24 = !DILocalVariable(name: "b", scope: !4, file: !5, line: 5, type: !9)
!25 = !DILocalVariable(name: "tmp", scope: !4, file: !5, line: 6, type: !9)
!26 = !{void (<2 x i32> addrspace(1)*, <2 x i32> addrspace(1)*, <2 x i32> addrspace(1)*)* @mul2, !27, !28, !29, !30, !31}
!27 = !{!"kernel_arg_addr_space", i32 1, i32 1, i32 1}
!28 = !{!"kernel_arg_access_qual", !"none", !"none", !"none"}
!29 = !{!"kernel_arg_type", !"int2*", !"int2*", !"int2*"}
!30 = !{!"kernel_arg_base_type", !"int __attribute__((ext_vector_type(2)))*", !"int __attribute__((ext_vector_type(2)))*", !"int __attribute__((ext_vector_type(2)))*"}
!31 = !{!"kernel_arg_type_qual", !"", !"", !""}
!32 = !{i32 2, !"Debug Info Version", i32 3}
!33 = !{!"clang version 3.8.0 "}
!34 = !DIExpression()
!35 = !DILocation(line: 1, scope: !4)
!36 = !DILocation(line: 3, scope: !4)
!37 = !DILocation(line: 4, scope: !4)
!38 = !DILocation(line: 5, scope: !4)
!39 = !DILocation(line: 6, scope: !4)
!40 = !DILocation(line: 7, scope: !4)
!41 = !DILocation(line: 8, scope: !4)

; Vectorized kernel function
; CHECK: @__vecz_v[[WIDTH:[0-9]+]]_mul2({{.*}} !dbg [[VECZ_SUBPROG:![0-9]+]]

; Check that intrinsics for user variable locations are still present
; CHECK-LT19: call void @llvm.dbg.value(metadata {{.*}} %in1, metadata [[DI_IN1:![0-9]+]], metadata [[EXPR:!DIExpression()]]
; CHECK-GE19: #dbg_value({{.*}} %in1, [[DI_IN1:![0-9]+]], [[EXPR:!DIExpression()]]
; CHECK-SAME: [[PARAM_LOC:![0-9]+]]

; CHECK-LT19: call void @llvm.dbg.value(metadata {{.*}} %in2, metadata [[DI_IN2:![0-9]+]], metadata [[EXPR]]
; CHECK-GE19: #dbg_value({{.*}} %in2, [[DI_IN2:![0-9]+]], [[EXPR]]
; CHECK-SAME: [[PARAM_LOC]]

; CHECK-LT19: call void @llvm.dbg.value(metadata {{.*}} %out, metadata [[DI_OUT:![0-9]+]], metadata [[EXPR]]
; CHECK-GE19: #dbg_value({{.*}} %out, [[DI_OUT:![0-9]+]], [[EXPR]]
; CHECK-SAME: [[PARAM_LOC]]

; CHECK-LT19: call void @llvm.dbg.value(metadata i64 %call, metadata [[DI_TID:![0-9]+]], metadata [[EXPR]]
; CHECK-GE19: #dbg_value(i64 %call, [[DI_TID:![0-9]+]], [[EXPR]]
; CHECK-SAME: [[TID_LOC:![0-9]+]]

; CHECK-LT19: call void @llvm.dbg.declare(metadata ptr %a, metadata [[DI_A:![0-9]+]], metadata [[EXPR]]
; CHECK-GE19: #dbg_declare(ptr %a, [[DI_A:![0-9]+]], [[EXPR]]
; CHECK-SAME: [[A_LOC:![0-9]+]]

; CHECK-LT19: call void @llvm.dbg.declare(metadata ptr %b, metadata [[DI_B:![0-9]+]], metadata [[EXPR]]
; CHECK-GE19: #dbg_declare(ptr %b, [[DI_B:![0-9]+]], [[EXPR]]
; CHECK-SAME: [[B_LOC:![0-9]+]]

; CHECK-LT19: call void @llvm.dbg.declare(metadata ptr %tmp, metadata [[DI_TMP:![0-9]+]], metadata [[EXPR]]
; CHECK-GE19: #dbg_declare(ptr %tmp, [[DI_TMP:![0-9]+]], [[EXPR]]
; CHECK-SAME: [[TMP_LOC:![0-9]+]]

; Debug info metadata entries
; CHECK:[[PTR_TYPE:![0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: [[DI_INT2:![0-9]+]], size: 64, align: 64)
; CHECK:[[DI_INT2]] = !DIDerivedType(tag: DW_TAG_typedef, name: "int2"

; CHECK: [[VECZ_SUBPROG]] = distinct !DISubprogram(name: "mul2"
; CHECK-SAME: retainedNodes: [[VECZ_VARS:![0-9]+]]

; CHECK: [[VECZ_VARS]] = !{[[DI_IN1]], [[DI_IN2]], [[DI_OUT]], [[DI_TID]], [[DI_A]], [[DI_B]], [[DI_TMP]]}

; CHECK: [[DI_IN1]] = !DILocalVariable(name: "in1", arg: 1, scope: [[VECZ_SUBPROG]],
; CHECK-SAME:line: 1, type: [[PTR_TYPE]]

; CHECK: [[DI_IN2]] = !DILocalVariable(name: "in2", arg: 2, scope: [[VECZ_SUBPROG]],
; CHECK-SAME:line: 1, type: [[PTR_TYPE]]

; CHECK: [[DI_OUT]] = !DILocalVariable(name: "out", arg: 3, scope: [[VECZ_SUBPROG]],
; CHECK-SAME: line: 1, type: [[PTR_TYPE]]

; CHECK: [[DI_TID]] = !DILocalVariable(name: "tid", scope: [[VECZ_SUBPROG]]
; CHECK-SAME:line: 3

; CHECK: [[DI_A]] = !DILocalVariable(name: "a", scope: [[VECZ_SUBPROG]],
; CHECK-SAME:line: 4

; CHECK: [[DI_B]] = !DILocalVariable(name: "b", scope: [[VECZ_SUBPROG]],
; CHECK-SAME: line: 5

; CHECK: [[DI_TMP]] = !DILocalVariable(name: "tmp", scope: [[VECZ_SUBPROG]],
; CHECK-SAME: line: 6
