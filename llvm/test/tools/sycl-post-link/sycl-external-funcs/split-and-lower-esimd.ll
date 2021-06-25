; RUN: sycl-post-link -split=auto -split-esimd -lower-esimd -O0 -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t_esimd_0.ll

; This test checks that unreferenced functions with sycl-module-id
; attribute are not dropped from the module and ESIMD lowering
; happens for them as well.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux-sycldevice"

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

define dso_local spir_func void @externalESIMDDeviceFunc() #0 !sycl_explicit_simd !0 {
entry:
  %0 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId to <3 x i64> addrspace(4)*), align 32
  %1 = extractelement <3 x i64> %0, i64 0
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }

!0 = !{}

; CHECK: define dso_local spir_func void @externalESIMDDeviceFunc()
; CHECK: entry:
; CHECK:   call <3 x i32> @llvm.genx.local.id.v3i32()
; CHECK:   call <3 x i32> @llvm.genx.local.size.v3i32()
; CHECK:   call i32 @llvm.genx.group.id.x()
; CHECK:   ret void
; CHECK: }
