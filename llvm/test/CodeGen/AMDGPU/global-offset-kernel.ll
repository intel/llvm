; RUN: opt -passes=globaloffset %s -S -o - | FileCheck %s

; The implicit offset argument of an AMDGPU kernel entry point must point into
; the kernel argument segment (addrspace(4)), which is where clang places
; `byref` kernel arguments. A generic pointer verifies only while the module
; carries a data layout whose alloca address space is not 0; tools that do not
; infer one from the triple, such as llvm-link during an RDC device link,
; reject it with "Calling convention disallows stack byref".

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

declare i64 @_Z27__spirv_BuiltInGlobalOffseti(i32)

declare void @use(i64)

define amdgpu_kernel void @_ZTS14example_kernel() {
; CHECK-LABEL: define amdgpu_kernel void @_ZTS14example_kernel() {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @use(i64 0)
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }
entry:
  %0 = call i64 @_Z27__spirv_BuiltInGlobalOffseti(i32 2)
  call void @use(i64 %0)
  ret void
}

; CHECK-LABEL: define amdgpu_kernel void @_ZTS14example_kernel_with_offset(
; CHECK-SAME:    ptr addrspace(4) byref([3 x i32]) [[OFFSET:%.*]])
; CHECK:         [[ALLOCA:%.*]] = alloca [3 x i32], align 4, addrspace(5)
; CHECK:         call void @llvm.memcpy{{.*}}(ptr addrspace(5) {{.*}}[[ALLOCA]], ptr addrspace(4) {{.*}}[[OFFSET]]

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"sycl-device", i32 1}
