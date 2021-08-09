; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv -spirv-text %t.bc -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.out.bc
; RUN: llvm-dis %t.out.bc -o - | FileCheck %s --check-prefix=CHECK-OCL-IR
; RUN: llvm-spirv -r %t.spv --spirv-target-env=SPV-IR -o %t.out.bc
; RUN: llvm-dis %t.out.bc -o - | FileCheck %s --check-prefix=CHECK-SPV-IR

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Check that builtin-call based SPV-IR is recognized by the translator.
; CHECK-SPIRV: Decorate [[Id:[0-9]+]] BuiltIn 28
; CHECK-SPIRV: Decorate [[Id:[0-9]+]] BuiltIn 34
; CHECK-SPIRV: Variable {{[0-9]+}} [[Id:[0-9]+]]
; CHECK-SPIRV: Variable {{[0-9]+}} [[Id:[0-9]+]]

; Function Attrs: nounwind readnone
define spir_kernel void @f() #0 !kernel_arg_addr_space !0 !kernel_arg_access_qual !0 !kernel_arg_type !0 !kernel_arg_base_type !0 !kernel_arg_type_qual !0 {
entry:
  %0 = call spir_func i32 @_Z29__spirv_BuiltInGlobalLinearIdv()
  ; CHECK-OCL-IR: call spir_func i32 @_Z20get_global_linear_idv()
  ; CHECK-SPV-IR: call spir_func i32 @_Z29__spirv_BuiltInGlobalLinearIdv()
  %1 = call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 1)
  ; CHECK-OCL-IR: call spir_func i64 @_Z13get_global_idj(i32 1)
  ; CHECK-SPV-IR: call spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32 1)
  ret void
}

declare spir_func i32 @_Z29__spirv_BuiltInGlobalLinearIdv()
declare spir_func i64 @_Z33__spirv_BuiltInGlobalInvocationIdi(i32)

attributes #0 = { alwaysinline nounwind readonly "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!0 = !{}
