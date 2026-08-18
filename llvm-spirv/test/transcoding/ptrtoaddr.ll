; Check "support" of ptrtoaddr instruction translation to SPIR-V.
; ptrtoaddr -> OpConvertPtrToU -> ptrtoint

; RUN: llvm-spirv %s -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %s -o %t.spv
; RUN: spirv-val %t.spv

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM
; RUN: %if spirv-backend %{ llc -O0 -mtriple=spirv64-unknown-unknown -filetype=obj %s -o %t.llc.spv %}
; RUN: %if spirv-backend %{ llvm-spirv -r %t.llc.spv -o %t.llc.rev.bc %}
; RUN: %if spirv-backend %{ llvm-dis %t.llc.rev.bc -o %t.llc.rev.ll %}
; RUN: %if spirv-backend %{ FileCheck %s --check-prefix=CHECK-LLVM < %t.llc.rev.ll %}

; CHECK-SPIRV: TypeInt [[#TypeInt:]] 64 0
; CHECK-SPIRV: ConvertPtrToU [[#TypeInt]] [[#]] [[#]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

define spir_kernel void @test_ptrtoaddr(ptr addrspace(1) %p, ptr addrspace(1) %res) {
entry:
  %addr = ptrtoaddr ptr addrspace(1) %p to i64
; CHECK-LLVM: %{{.*}} = ptrtoint ptr addrspace(1) %{{.*}} to i64
  store i64 %addr, ptr addrspace(1) %res, align 8
  ret void
}

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
