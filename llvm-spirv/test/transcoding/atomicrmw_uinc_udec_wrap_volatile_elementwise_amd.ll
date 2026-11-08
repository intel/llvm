; Check that volatile and elementwise are lowered to the last two args of the
; OpFunctionCall.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r --spirv-target-triple=spirv64-amd-amdhsa %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM
; RUN: spirv-val %t.spv

; CHECK-SPIRV-DAG: Decorate [[#UIncFn:]] LinkageAttributes "__translate_spirv_atomic_uinc_wrap_p1_v2i32" Import
; CHECK-SPIRV-DAG: Decorate [[#UDecFn:]] LinkageAttributes "__translate_spirv_atomic_udec_wrap_p1_v2i32" Import

; CHECK-SPIRV-DAG: TypeInt [[#Int:]] 32 0
; CHECK-SPIRV-DAG: TypeVector [[#V2I32:]] [[#Int]] 2
; CHECK-SPIRV-DAG: TypeBool [[#Bool:]]
; CHECK-SPIRV-DAG: ConstantFalse [[#Bool]] [[#False:]]
; CHECK-SPIRV-DAG: ConstantTrue [[#Bool]] [[#True:]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv64-amd-amdhsa"

@uv2i32 = common dso_local addrspace(1) global <2 x i32> zeroinitializer, align 8

; CHECK-SPIRV: FunctionCall [[#V2I32]] [[#Res:]] [[#UIncFn]] [[#]] [[#]] [[#]] [[#]] [[#False]] [[#True]]
; CHECK-SPIRV-NEXT: ReturnValue [[#Res]]
; CHECK-LLVM: %[[#R:]] = atomicrmw elementwise uinc_wrap ptr addrspace(1) @uv2i32, <2 x i32> splat (i32 42) monotonic
; CHECK-LLVM-NEXT: ret <2 x i32> %[[#R]]
define dso_local spir_func <2 x i32> @elementwise_uinc() local_unnamed_addr {
entry:
  %r = atomicrmw elementwise uinc_wrap ptr addrspace(1) @uv2i32, <2 x i32> splat (i32 42) monotonic
  ret <2 x i32> %r
}

; CHECK-SPIRV: FunctionCall [[#V2I32]] [[#Res:]] [[#UDecFn]] [[#]] [[#]] [[#]] [[#]] [[#True]] [[#True]]
; CHECK-SPIRV-NEXT: ReturnValue [[#Res]]
; CHECK-LLVM: %[[#R:]] = atomicrmw volatile elementwise udec_wrap ptr addrspace(1) @uv2i32, <2 x i32> splat (i32 42) acquire
; CHECK-LLVM-NEXT: ret <2 x i32> %[[#R]]
define dso_local spir_func <2 x i32> @elementwise_volatile_udec() local_unnamed_addr {
entry:
  %r = atomicrmw volatile elementwise udec_wrap ptr addrspace(1) @uv2i32, <2 x i32> splat (i32 42) acquire
  ret <2 x i32> %r
}

; CHECK-SPIRV: FunctionCall [[#V2I32]] [[#Res:]] [[#UIncFn]] [[#]] [[#]] [[#]] [[#]] [[#False]] [[#False]]
; CHECK-SPIRV-NEXT: ReturnValue [[#Res]]
; CHECK-LLVM: %[[#R:]] = atomicrmw uinc_wrap ptr addrspace(1) @uv2i32, <2 x i32> splat (i32 42) monotonic
; CHECK-LLVM-NEXT: ret <2 x i32> %[[#R]]
define dso_local spir_func <2 x i32> @whole_vector_uinc() local_unnamed_addr {
entry:
  %r = atomicrmw uinc_wrap ptr addrspace(1) @uv2i32, <2 x i32> splat (i32 42) monotonic
  ret <2 x i32> %r
}

; CHECK-SPIRV: FunctionCall [[#V2I32]] [[#Res:]] [[#UIncFn]] [[#]] [[#]] [[#]] [[#]] [[#True]] [[#False]]
; CHECK-SPIRV-NEXT: ReturnValue [[#Res]]
; CHECK-LLVM: %[[#R:]] = atomicrmw volatile uinc_wrap ptr addrspace(1) @uv2i32, <2 x i32> splat (i32 42) monotonic
; CHECK-LLVM-NEXT: ret <2 x i32> %[[#R]]
define dso_local spir_func <2 x i32> @whole_vector_volatile_uinc() local_unnamed_addr {
entry:
  %r = atomicrmw volatile uinc_wrap ptr addrspace(1) @uv2i32, <2 x i32> splat (i32 42) monotonic
  ret <2 x i32> %r
}