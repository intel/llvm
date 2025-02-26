; RUN: sycl-post-link -properties -split=auto < %s -o %t.files.table
; RUN: FileCheck %s -input-file=%t.files_0.prop

; CHECK:[SYCL/devicelib req mask]
; CHECK: DeviceLibReqMask=1|64

source_filename = "main.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spirv64-unknown-unknown"

declare spir_func i32 @__devicelib_imf_umulhi(i32 noundef %0, i32 noundef %1)

; Function Attrs: convergent mustprogress noinline norecurse optnone
define weak_odr dso_local spir_kernel void @kernel() #0 {
entry:
  %0 = call i32 @__devicelib_imf_umulhi(i32 0, i32 0)
  ret void
}

attributes #0 = { "sycl-module-id"="main.cpp" }
