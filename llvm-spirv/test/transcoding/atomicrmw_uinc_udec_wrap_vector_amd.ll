; Check that uinc_wrap/udec_wrap are properly lowered for different vector
; types.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r --spirv-target-triple=spirv64-amd-amdhsa %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM
; RUN: spirv-val %t.spv

; CHECK-SPIRV-DAG: Decorate [[#UIncFnV2I32:]] LinkageAttributes "__translate_spirv_atomic_uinc_wrap_p1_v2i32" Import
; CHECK-SPIRV-DAG: Decorate [[#UDecFnV4I16:]] LinkageAttributes "__translate_spirv_atomic_udec_wrap_p1_v4i16" Import
; CHECK-SPIRV-DAG: Decorate [[#UIncFnV8I8:]] LinkageAttributes "__translate_spirv_atomic_uinc_wrap_p1_v8i8" Import

; CHECK-SPIRV-DAG: TypeInt [[#Int:]] 32 0
; CHECK-SPIRV-DAG: TypeInt [[#Short:]] 16 0
; CHECK-SPIRV-DAG: TypeInt [[#Char:]] 8 0
; CHECK-SPIRV-DAG: TypeVector [[#V2I32:]] [[#Int]] 2
; CHECK-SPIRV-DAG: TypeVector [[#V4I16:]] [[#Short]] 4
; CHECK-SPIRV-DAG: TypeVector [[#V8I8:]] [[#Char]] 8
; CHECK-SPIRV-DAG: Constant [[#Int]] [[#Scope_CrossDevice:]] 0{{ *$}}
; CHECK-SPIRV-DAG: Constant [[#Int]] [[#MemSem_SequentiallyConsistent:]] 528{{ *$}}
; CHECK-SPIRV-DAG: TypeBool [[#Bool:]]
; CHECK-SPIRV-DAG: ConstantFalse [[#Bool]] [[#False:]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv64-amd-amdhsa"

@uv2i32 = common dso_local addrspace(1) global <2 x i32> zeroinitializer, align 8
@uv4i16 = common dso_local addrspace(1) global <4 x i16> zeroinitializer, align 8
@uv8i8 = common dso_local addrspace(1) global <8 x i8> zeroinitializer, align 8

; CHECK-SPIRV: FunctionCall [[#V2I32]] [[#Res:]] [[#UIncFnV2I32]] [[#]] [[#Scope_CrossDevice]] [[#MemSem_SequentiallyConsistent]] [[#]] [[#False]] [[#False]]
; CHECK-SPIRV-NEXT: ReturnValue [[#Res]]
; CHECK-LLVM: %[[#R:]] = atomicrmw uinc_wrap ptr addrspace(1) @uv2i32, <2 x i32> splat (i32 42) seq_cst, align 8
; CHECK-LLVM-NEXT: ret <2 x i32> %[[#R]]
define dso_local spir_func <2 x i32> @uinc_wrap_v2i32() local_unnamed_addr {
entry:
  %r = atomicrmw uinc_wrap ptr addrspace(1) @uv2i32, <2 x i32> splat (i32 42) seq_cst
  ret <2 x i32> %r
}

; CHECK-SPIRV: FunctionCall [[#V4I16]] [[#Res:]] [[#UDecFnV4I16]] [[#]] [[#Scope_CrossDevice]] [[#MemSem_SequentiallyConsistent]] [[#]] [[#False]] [[#False]]
; CHECK-SPIRV-NEXT: ReturnValue [[#Res]]
; CHECK-LLVM: %[[#R:]] = atomicrmw udec_wrap ptr addrspace(1) @uv4i16, <4 x i16> splat (i16 42) seq_cst, align 8
; CHECK-LLVM-NEXT: ret <4 x i16> %[[#R]]
define dso_local spir_func <4 x i16> @udec_wrap_v4i16() local_unnamed_addr {
entry:
  %r = atomicrmw udec_wrap ptr addrspace(1) @uv4i16, <4 x i16> splat (i16 42) seq_cst
  ret <4 x i16> %r
}

; CHECK-SPIRV: FunctionCall [[#V8I8]] [[#Res:]] [[#UIncFnV8I8]] [[#]] [[#Scope_CrossDevice]] [[#MemSem_SequentiallyConsistent]] [[#]] [[#False]] [[#False]]
; CHECK-SPIRV-NEXT: ReturnValue [[#Res]]
; CHECK-LLVM: %[[#R:]] = atomicrmw uinc_wrap ptr addrspace(1) @uv8i8, <8 x i8> splat (i8 42) seq_cst, align 8
; CHECK-LLVM-NEXT: ret <8 x i8> %[[#R]]
define dso_local spir_func <8 x i8> @uinc_wrap_v8i8() local_unnamed_addr {
entry:
  %r = atomicrmw uinc_wrap ptr addrspace(1) @uv8i8, <8 x i8> splat (i8 42) seq_cst
  ret <8 x i8> %r
}
