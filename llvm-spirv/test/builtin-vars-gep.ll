; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -r -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM

source_filename = "builtin-vars-gep.ll"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64"

@__spirv_BuiltInWorkgroupSize = external addrspace(1) constant <3 x i64>, align 32

; Function Attrs: alwaysinline convergent nounwind mustprogress
define spir_func void @foo() {
entry:
  %GroupID = alloca [3 x i64], align 8
  %0 = addrspacecast ptr addrspace(1) @__spirv_BuiltInWorkgroupSize to ptr addrspace(4)
; CHECK-LLVM: %[[GLocalSize0:[0-9]+]] = call spir_func i64 @_Z14get_local_sizej(i32 0) #1
  %1 = addrspacecast ptr addrspace(1) @__spirv_BuiltInWorkgroupSize to ptr addrspace(4)
  %2 = getelementptr <3 x i64>, ptr addrspace(4) %1, i64 0, i64 2
  %3 = load i64, ptr addrspace(4) %0, align 32
  %4 = load i64, ptr addrspace(4) %2, align 8
; CHECK-LLVM: %[[GLocalSize2:[0-9]+]] = call spir_func i64 @_Z14get_local_sizej(i32 2) #1
; CHECK-LLVM:  mul i64 %[[GLocalSize0]], %[[GLocalSize2]]
  %mul = mul i64 %3, %4
  ret void
}

; Function Attrs: alwaysinline convergent nounwind mustprogress
define spir_func void @foo_i8gep() {
entry:
  %GroupID = alloca [3 x i64], align 8
  %0 = addrspacecast ptr addrspace(1) @__spirv_BuiltInWorkgroupSize to ptr addrspace(4)
; CHECK-LLVM: %[[GLocalSize0:[0-9]+]] = call spir_func i64 @_Z14get_local_sizej(i32 0) #1
  %1 = addrspacecast ptr addrspace(1) @__spirv_BuiltInWorkgroupSize to ptr addrspace(4)
  %2 = getelementptr i8, ptr addrspace(4) %1, i64 16
  %3 = load i64, ptr addrspace(4) %0, align 32
  %4 = load i64, ptr addrspace(4) %2, align 8
; CHECK-LLVM: %[[GLocalSize2:[0-9]+]] = call spir_func i64 @_Z14get_local_sizej(i32 2) #1
; CHECK-LLVM:  mul i64 %[[GLocalSize0]], %[[GLocalSize2]]
  %mul = mul i64 %3, %4
  ret void
}
