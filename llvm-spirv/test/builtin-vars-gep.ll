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
  %0 = addrspacecast <3 x i64> addrspace(1)* @__spirv_BuiltInWorkgroupSize to <3 x i64> addrspace(4)*
  %1 = getelementptr <3 x i64>, <3 x i64> addrspace(4)* %0, i64 0, i64 0
; CHECK-LLVM: %[[GLocalSize0:[0-9]+]] = call spir_func i64 @_Z14get_local_sizej(i32 0) #1
; CHECK-LLVM: %[[Ins0:[0-9]+]] = insertelement <3 x i64> undef, i64 %[[GLocalSize0]], i32 0
; CHECK-LLVM: %[[GLocalSize1:[0-9]+]] = call spir_func i64 @_Z14get_local_sizej(i32 1) #1
; CHECK-LLVM: %[[Ins1:[0-9]+]] = insertelement <3 x i64> %[[Ins0]], i64 %[[GLocalSize1]], i32 1
; CHECK-LLVM: %[[GLocalSize2:[0-9]+]] = call spir_func i64 @_Z14get_local_sizej(i32 2) #1
; CHECK-LLVM: %[[Ins2:[0-9]+]] = insertelement <3 x i64> %[[Ins1]], i64 %[[GLocalSize2]], i32 2
; CHECK-LLVM: %[[Extract:[0-9]+]] = extractelement <3 x i64> %[[Ins2]], i64 0
  %2 = addrspacecast <3 x i64> addrspace(1)* @__spirv_BuiltInWorkgroupSize to <3 x i64> addrspace(4)*
  %3 = getelementptr <3 x i64>, <3 x i64> addrspace(4)* %2, i64 0, i64 2
  %4 = load i64, i64 addrspace(4)* %1, align 32
  %5 = load i64, i64 addrspace(4)* %3, align 8
; CHECK-LLVM: %[[GLocalSize01:[0-9]+]] = call spir_func i64 @_Z14get_local_sizej(i32 0) #1
; CHECK-LLVM: %[[Ins01:[0-9]+]] = insertelement <3 x i64> undef, i64 %[[GLocalSize01]], i32 0
; CHECK-LLVM: %[[GLocalSize11:[0-9]+]] = call spir_func i64 @_Z14get_local_sizej(i32 1) #1
; CHECK-LLVM: %[[Ins11:[0-9]+]] = insertelement <3 x i64> %[[Ins01]], i64 %[[GLocalSize11]], i32 1
; CHECK-LLVM: %[[GLocalSize21:[0-9]+]] = call spir_func i64 @_Z14get_local_sizej(i32 2) #1
; CHECK-LLVM: %[[Ins21:[0-9]+]] = insertelement <3 x i64> %[[Ins11]], i64 %[[GLocalSize21]], i32 2
; CHECK-LLVM: %[[Extract1:[0-9]+]] = extractelement <3 x i64> %[[Ins21]], i64 2
; CHECK-LLVM:  mul i64 %[[Extract]], %[[Extract1]]
  %mul = mul i64 %4, %5
  ret void
}

