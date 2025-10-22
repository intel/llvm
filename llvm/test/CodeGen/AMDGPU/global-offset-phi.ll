; RUN: opt -bugpoint-enable-legacy-pm -globaloffset %s -S -o - | FileCheck %s

; Check that phi is correctly handled in load's defs collection.

; CHECK-NOT: call ptr addrspace(5) @llvm.amdgcn.implicit.offset()

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

define internal i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 %0) {
  switch i32 %0, label %14 [
    i32 0, label %2
    i32 1, label %4
    i32 2, label %7
  ]

2:                                                ; preds = %1
  %3 = tail call ptr addrspace(5) @llvm.amdgcn.implicit.offset()
  br label %10

4:                                                ; preds = %1
  %5 = tail call ptr addrspace(5) @llvm.amdgcn.implicit.offset()
  %6 = getelementptr inbounds nuw i8, ptr addrspace(5) %5, i32 4
  br label %10

7:                                                ; preds = %1
  %8 = tail call ptr addrspace(5) @llvm.amdgcn.implicit.offset()
  %9 = getelementptr inbounds nuw i8, ptr addrspace(5) %8, i32 8
  br label %10

10:                                               ; preds = %7, %4, %2
  %11 = phi ptr addrspace(5) [ %3, %2 ], [ %6, %4 ], [ %9, %7 ]
  %12 = load i32, ptr addrspace(5) %11, align 4
  %13 = load i32, ptr addrspace(5) %11, align 4
  ret i64 0

14:                                               ; preds = %1
  unreachable
}

declare ptr addrspace(5) @llvm.amdgcn.implicit.offset()

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"sycl-device", i32 1}
