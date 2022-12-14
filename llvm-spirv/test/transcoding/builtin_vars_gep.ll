; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r -emit-opaque-pointers %t.spv -o %t.out.bc
; RUN: llvm-dis %t.out.bc -o - | FileCheck %s --check-prefix=CHECK-OCL-IR
; RUN: llvm-spirv -r -emit-opaque-pointers %t.spv --spirv-target-env=SPV-IR -o %t.out.bc
; RUN: llvm-dis %t.out.bc -o - | FileCheck %s --check-prefix=CHECK-SPV-IR

; Check that produced builtin-call-based SPV-IR is recognized by the translator
; RUN: llvm-spirv -spirv-text %t.out.bc -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: Decorate [[Id:[0-9]+]] BuiltIn 28
; CHECK-SPIRV: Variable {{[0-9]+}} [[Id:[0-9]+]]

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; Function Attrs: nounwind readnone
define spir_kernel void @f() {
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32
  ; CHECK-OCL-IR: %[[#ID1:]] = call spir_func i64 @_Z13get_global_idj(i32 0)
  ; CHECK-OCL-IR: %[[#VEC1:]] = insertelement <3 x i64> undef, i64 %[[#ID1]], i32 0
  ; CHECK-OCL-IR: %[[#ID2:]] = call spir_func i64 @_Z13get_global_idj(i32 1)
  ; CHECK-OCL-IR: %[[#VEC2:]] = insertelement <3 x i64> %[[#VEC1]], i64 %[[#ID2]], i32 1
  ; CHECK-OCL-IR: %[[#ID3:]] = call spir_func i64 @_Z13get_global_idj(i32 2)
  ; CHECK-OCL-IR: %[[#VEC3:]] = insertelement <3 x i64> %[[#VEC2]], i64 %[[#ID3]], i32 2
  ; CHECK-OCL-IR: %[[#]] = extractelement <3 x i64> %[[#VEC3]], i32 0

  ; CHECK-SPV-IR: %[[#ID1:]] = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 0)
  ; CHECK-SPV-IR: %[[#VEC1:]] = insertelement <3 x i64> undef, i64 %[[#ID1]], i32 0
  ; CHECK-SPV-IR: %[[#ID2:]] = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 1)
  ; CHECK-SPV-IR: %[[#VEC2:]] = insertelement <3 x i64> %[[#VEC1]], i64 %[[#ID2]], i32 1
  ; CHECK-SPV-IR: %[[#ID3:]] = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 2)
  ; CHECK-SPV-IR: %[[#VEC3:]] = insertelement <3 x i64> %[[#VEC2]], i64 %[[#ID3]], i32 2
  ; CHECK-SPV-IR: %[[#]] = extractelement <3 x i64> %[[#VEC3]], i32 0

  ret void
}
