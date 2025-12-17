; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o - -spirv-text | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r --spirv-target-env=SPV-IR %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-SPV-IR
; RUN: llvm-spirv -r  %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-OCL-IR

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir-unknown-unknown"

; CHECK-SPIRV-DAG: TypeEvent [[#EventTy:]]
; CHECK-SPIRV-DAG: TypeStruct [[#StructEventTy:]] [[#EventTy]]
; CHECK-SPIRV-DAG: TypePointer [[#FunPtrStructEventTy:]] 7 [[#StructEventTy]]
; CHECK-SPIRV-DAG: TypePointer [[#GenPtrEventTy:]] 8 [[#StructEventTy]]
; CHECK-SPIRV-DAG: TypePointer [[#FunPtrEventTy:]] 8 [[#EventTy]]
; CHECK-SPIRV: Function
; CHECK-SPIRV: Variable [[#FunPtrStructEventTy]] [[#Var:]] 7
; CHECK-SPIRV-NEXT:  PtrCastToGeneric [[#GenPtrEventTy]] [[#GenEvent:]] [[#Var]]
; CHECK-SPIRV-NEXT:  Bitcast [[#FunPtrEventTy]] [[#FunEvent:]] [[#GenEvent]]
; CHECK-SPIRV-NEXT:  GroupWaitEvents [[#]] [[#]] [[#FunEvent]]

; CHECK-SPV-IR: __spirv_GroupWaitEvents
; CHECK-OCL-IR: spir_func void @_Z17wait_group_events
; TODO: The call to @_Z17wait_group_events is not yet lowered to the corresponding SPIR-V instruction.
;       It currently remains as a plain function call during SPIR-V generation.
%"class.sycl::_V1::device_event" = type { target("spirv.Event") }

define weak_odr dso_local spir_kernel void @foo() {
entry:
  %var = alloca %"class.sycl::_V1::device_event"
  %eventptr = addrspacecast ptr %var to ptr addrspace(4)
  call spir_func void @_Z23__spirv_GroupWaitEventsjiP9ocl_event(i32 2, i32 1, ptr addrspace(4) %eventptr)
  ret void
}

declare dso_local spir_func void @_Z23__spirv_GroupWaitEventsjiP9ocl_event(i32, i32, ptr addrspace(4))
