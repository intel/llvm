; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv --to-text %t.spv -o %t
; RUN: llvm-spirv -r %t -spirv-text --spirv-target-env=SPV-IR --spirv-builtin-format=function -o %t2_rev.bc
; RUN: llvm-spirv -r %t -spirv-text --spirv-target-env=SPV-IR --spirv-builtin-format=global -o %t3_rev.bc
; RUN: llvm-spirv -r %t -spirv-text --spirv-builtin-format=function -o %t2_rev_ocl.bc
; RUN: llvm-dis < %t2_rev.bc | FileCheck --check-prefix=CHECK-FUNCTION-FORMAT-REV %s
; RUN: llvm-dis < %t3_rev.bc | FileCheck --check-prefix=CHECK-GLOBAL-FORMAT-REV %s
; RUN: llvm-dis < %t2_rev_ocl.bc | FileCheck --check-prefix=CHECK-FUNCTION-FORMAT-OCL-REV %s
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.bc --spirv-builtin-format=function -o %t2.spv
; RUN: spirv-val %t2.spv
; RUN: llvm-spirv %t.bc --spirv-builtin-format=global -o %t3.spv
; RUN: spirv-val %t3.spv

; CHECK-FUNCTION-FORMAT-REV: declare spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32)
; CHECK-FUNCTION-FORMAT-OCL-REV: declare spir_func i64 @_Z12get_group_idj(i32)

; CHECK-GLOBAL-FORMAT-REV: @__spirv_BuiltInWorkgroupId = external addrspace(7) constant <3 x i64>

; ModuleID = 'test.bc'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Function Attrs: nounwind
declare spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32) #0

; Function Attrs: convergent norecurse nounwind
define weak_odr dso_local spir_kernel void @foo() {
entry:
  %0 = call spir_func i64 @_Z26__spirv_BuiltInWorkgroupIdi(i32 0) #0
 ret void
}
