; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv -spirv-ext=+SPV_INTEL_vector_compute
; RUN: llvm-spirv -r %t.spv --spirv-target-env=SPV-IR -o %t.out.bc
; RUN: llvm-dis %t.out.bc -o - | FileCheck %s --check-prefix=CHECK-SPV-IR

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; Function Attrs: nounwind readnone
define spir_kernel void @f() {
entry:
  %0 = load <6 x i32>, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, align 32
  %1 = extractelement <6 x i32> %0, i64 0
  %2 = add i32 5, %1
  ret void
; CHECK-SPV-IR: %[[#ID0:]] = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 0) #1
; CHECK-SPV-IR: %[[#ID1:]] = insertelement <3 x i64> undef, i64 %[[#ID0]], i32 0
; CHECK-SPV-IR: %[[#ID2:]] = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 1) #1
; CHECK-SPV-IR: %[[#ID3:]] = insertelement <3 x i64> %[[#ID1]], i64 %[[#ID2]], i32 1
; CHECK-SPV-IR: %[[#ID4:]] = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 2) #1
; CHECK-SPV-IR: %[[#ID5:]] = insertelement <3 x i64> %[[#ID3]], i64 %[[#ID4]], i32 2
; CHECK-SPV-IR: %[[#ID6:]] = bitcast <3 x i64> %[[#ID5]] to <6 x i32>
; CHECK-SPV-IR: %[[#ID7:]] = extractelement <6 x i32> %[[#ID6]], i32 0
; CHECK-SPV-IR: = add i32 5, %[[#ID7]]
}
