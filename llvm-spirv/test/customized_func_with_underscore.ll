; RUN: llvm-as %s -o - | llvm-spirv -o %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s
; RUN: %if spirv-backend %{ llc -O0 -mtriple=spirv64-unknown-unknown -filetype=obj %s -o %t.llc.spv %}
; RUN: %if spirv-backend %{ llvm-spirv -r %t.llc.spv -o %t.llc.rev.bc %}
; RUN: %if spirv-backend %{ llvm-dis %t.llc.rev.bc -o %t.llc.rev.ll %}
; RUN: %if spirv-backend %{ FileCheck %s < %t.llc.rev.ll %}

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

; Function Attrs: nounwind readnone
define spir_kernel void @f() #0 !kernel_arg_addr_space !0 !kernel_arg_access_qual !0 !kernel_arg_type !0 !kernel_arg_base_type !0 !kernel_arg_type_qual !0 {
entry:
  %0 = call spir_func i32 @_Z32__spirv_somefunc_with_underscorev()
  ; CHECK: call spir_func i32 @_Z32__spirv_somefunc_with_underscorev()
  %1 = call spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()
  ; CHECK: call spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()
  ret void
}

declare spir_func i32 @_Z32__spirv_somefunc_with_underscorev()
declare spir_func i64 @_Z28__spirv_GlobalInvocationId_xv()

attributes #0 = { nounwind readnone }

!0 = !{}
