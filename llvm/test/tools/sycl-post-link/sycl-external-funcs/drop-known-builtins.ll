; RUN: sycl-post-link -ir-output-only -split=auto -S %s -o %t.ll
; RUN: FileCheck %s -input-file=%t.ll

; This test checks that known SPIRV and SYCL builtin functions
; (that are also marked with SYCL_EXTRENAL) are not considered
; as module entry points and thus are not added as entry to the
; device binary symbol table. So, they can be dropped if
; unreferenced.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-linux-sycldevice"

define dso_local spir_func void @_Z28__spirv_GlobalInvocationId_xv() #0 {
  ret void
}
define dso_local spir_func void @_Z28__spXrv_GlobalInvocationId_xv() #0 {
  ret void
}

define dso_local spir_func void @_Z33__sycl_getScalarSpecConstantValue() #0 {
  ret void
}
define dso_local spir_func void @_Z33__sXcl_getScalarSpecConstantValue() #0 {
  ret void
}

attributes #0 = { "sycl-module-id"="a.cpp" }

; CHECK-NOT: define dso_local spir_func void @_Z28__spirv_GlobalInvocationId_xv()
; CHECK-NOT: define dso_local spir_func void @_Z33__sycl_getScalarSpecConstantValue()

; CHECK-DAG: define dso_local spir_func void @_Z28__spXrv_GlobalInvocationId_xv()
; CHECK-DAG: define dso_local spir_func void @_Z33__sXcl_getScalarSpecConstantValue()
