; Check that uinc_wrap/udec_wrap is properly translated when targeting AMD
; triples. These should be lowered into a function call to an imported
; __translate_spirv_atomic_{uinc/udec}_wrap_p{AS}_{datatype} with args being
; (Ptr, MemScope, MemSem, Value, IsVolatile, IsElementwise).

; RUN: llvm-spirv %s -o %t.spv
; RUN: llvm-spirv -to-text %t.spv -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r --spirv-target-triple=spirv64-amd-amdhsa %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefix=CHECK-LLVM
; RUN: spirv-val %t.spv

; CHECK-SPIRV-DAG: Decorate [[#UIncFn:]] LinkageAttributes "__translate_spirv_atomic_uinc_wrap_p1_i32" Import
; CHECK-SPIRV-DAG: Decorate [[#UDecFn:]] LinkageAttributes "__translate_spirv_atomic_udec_wrap_p1_i32" Import
; CHECK-SPIRV-DAG: Decorate [[#UIncFnLocal:]] LinkageAttributes "__translate_spirv_atomic_uinc_wrap_p3_i32" Import
; CHECK-SPIRV-DAG: Decorate [[#UIncFn64:]] LinkageAttributes "__translate_spirv_atomic_uinc_wrap_p1_i64" Import

; CHECK-SPIRV: TypeInt [[#Int:]] 32 0
; CHECK-SPIRV-DAG: TypeInt [[#Long:]] 64 0
; CHECK-SPIRV-DAG: Constant [[#Int]] [[#Scope_CrossDevice:]] 0{{ *$}}
; CHECK-SPIRV-DAG: Constant [[#Int]] [[#Scope_Device:]] 1{{ *$}}
; CHECK-SPIRV-DAG: Constant [[#Int]] [[#Scope_Workgroup:]] 2{{ *$}}
; CHECK-SPIRV-DAG: Constant [[#Int]] [[#Scope_Subgroup:]] 3{{ *$}}
; CHECK-SPIRV-DAG: Constant [[#Int]] [[#Scope_Invocation:]] 4{{ *$}}
; CHECK-SPIRV-DAG: Constant [[#Int]] [[#MemSem_Relaxed:]] 512{{ *$}}
; CHECK-SPIRV-DAG: Constant [[#Int]] [[#MemSem_Acquire:]] 514{{ *$}}
; CHECK-SPIRV-DAG: Constant [[#Int]] [[#MemSem_Release:]] 516{{ *$}}
; CHECK-SPIRV-DAG: Constant [[#Int]] [[#MemSem_AcquireRelease:]] 520{{ *$}}
; CHECK-SPIRV-DAG: Constant [[#Int]] [[#MemSem_SequentiallyConsistent:]] 528{{ *$}}
; CHECK-SPIRV-DAG: Constant [[#Int]] [[#Value:]] 42{{ *$}}
; The storage class contributes to the memory semantics: a Workgroup pointer
; carries WorkgroupMemory (256), a CrossWorkgroup one CrossWorkgroupMemory (512).
; CHECK-SPIRV-DAG: Constant [[#Int]] [[#MemSem_Relaxed_Local:]] 256{{ *$}}
; CHECK-SPIRV-DAG: Constant [[#Long]] [[#Value64:]] 42 0{{ *$}}
; CHECK-SPIRV-DAG: TypeBool [[#Bool:]]
; CHECK-SPIRV-DAG: ConstantFalse [[#Bool]] [[#False:]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv64-amd-amdhsa"

@ui = common dso_local addrspace(1) global i32 0, align 4
@lui = common dso_local addrspace(3) global i32 0, align 4
@ul = common dso_local addrspace(1) global i64 0, align 8

; CHECK-SPIRV: FunctionCall [[#Int]] [[#Res:]] [[#UIncFn]] [[#]] [[#Scope_CrossDevice]] [[#MemSem_Relaxed]] [[#Value]] [[#False]] [[#False]]
; CHECK-SPIRV-NEXT: ReturnValue [[#Res]]
; CHECK-LLVM: %[[#R:]] = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 monotonic
; CHECK-LLVM-NEXT: ret i32 %[[#R]]
define spir_func i32 @test_uinc_wrap_monotonic() {
entry:
  %r = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 monotonic
  ret i32 %r
}

; CHECK-SPIRV: FunctionCall [[#Int]] [[#Res:]] [[#UIncFn]] [[#]] [[#Scope_CrossDevice]] [[#MemSem_Acquire]] [[#Value]] [[#False]] [[#False]]
; CHECK-SPIRV-NEXT: ReturnValue [[#Res]]
; CHECK-LLVM: %[[#R:]] = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 acquire
; CHECK-LLVM-NEXT: ret i32 %[[#R]]
define spir_func i32 @test_uinc_wrap_acquire() {
entry:
  %r = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 acquire
  ret i32 %r
}

; CHECK-SPIRV: FunctionCall [[#Int]] [[#Res:]] [[#UIncFn]] [[#]] [[#Scope_CrossDevice]] [[#MemSem_Release]] [[#Value]] [[#False]] [[#False]]
; CHECK-SPIRV-NEXT: ReturnValue [[#Res]]
; CHECK-LLVM: %[[#R:]] = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 release
; CHECK-LLVM-NEXT: ret i32 %[[#R]]
define spir_func i32 @test_uinc_wrap_release() {
entry:
  %r = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 release
  ret i32 %r
}

; CHECK-SPIRV: FunctionCall [[#Int]] [[#Res:]] [[#UIncFn]] [[#]] [[#Scope_CrossDevice]] [[#MemSem_AcquireRelease]] [[#Value]] [[#False]] [[#False]]
; CHECK-SPIRV-NEXT: ReturnValue [[#Res]]
; CHECK-LLVM: %[[#R:]] = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 acq_rel
; CHECK-LLVM-NEXT: ret i32 %[[#R]]
define spir_func i32 @test_uinc_wrap_acq_rel() {
entry:
  %r = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 acq_rel
  ret i32 %r
}

; CHECK-SPIRV: FunctionCall [[#Int]] [[#Res:]] [[#UDecFn]] [[#]] [[#Scope_CrossDevice]] [[#MemSem_SequentiallyConsistent]] [[#Value]] [[#False]] [[#False]]
; CHECK-SPIRV-NEXT: ReturnValue [[#Res]]
; CHECK-LLVM: %[[#R:]] = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 seq_cst
; CHECK-LLVM-NEXT: ret i32 %[[#R]]
define spir_func i32 @test_udec_wrap_seq_cst() {
entry:
  %r = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 seq_cst
  ret i32 %r
}

; The reverse translation picks the generic LLVM scope names for a non-AMDGCN
; target triple.

; CHECK-SPIRV: FunctionCall [[#Int]] [[#Res:]] [[#UDecFn]] [[#]] [[#Scope_Device]] [[#MemSem_SequentiallyConsistent]] [[#Value]] [[#False]] [[#False]]
; CHECK-SPIRV-NEXT: ReturnValue [[#Res]]
; CHECK-LLVM: %[[#R:]] = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 syncscope("device") seq_cst
; CHECK-LLVM-NEXT: ret i32 %[[#R]]
define spir_func i32 @test_udec_wrap_device_scope() {
entry:
  %r = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 syncscope("device") seq_cst
  ret i32 %r
}

; CHECK-SPIRV: FunctionCall [[#Int]] [[#Res:]] [[#UDecFn]] [[#]] [[#Scope_Workgroup]] [[#MemSem_SequentiallyConsistent]] [[#Value]] [[#False]] [[#False]]
; CHECK-SPIRV-NEXT: ReturnValue [[#Res]]
; CHECK-LLVM: %[[#R:]] = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 syncscope("workgroup") seq_cst
; CHECK-LLVM-NEXT: ret i32 %[[#R]]
define spir_func i32 @test_udec_wrap_workgroup_scope() {
entry:
  %r = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 syncscope("workgroup") seq_cst
  ret i32 %r
}

; CHECK-SPIRV: FunctionCall [[#Int]] [[#Res:]] [[#UDecFn]] [[#]] [[#Scope_Subgroup]] [[#MemSem_SequentiallyConsistent]] [[#Value]] [[#False]] [[#False]]
; CHECK-SPIRV-NEXT: ReturnValue [[#Res]]
; CHECK-LLVM: %[[#R:]] = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 syncscope("subgroup") seq_cst
; CHECK-LLVM-NEXT: ret i32 %[[#R]]
define spir_func i32 @test_udec_wrap_subgroup_scope() {
entry:
  %r = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 syncscope("subgroup") seq_cst
  ret i32 %r
}

; CHECK-SPIRV: FunctionCall [[#Int]] [[#Res:]] [[#UDecFn]] [[#]] [[#Scope_Invocation]] [[#MemSem_SequentiallyConsistent]] [[#Value]] [[#False]] [[#False]]
; CHECK-SPIRV-NEXT: ReturnValue [[#Res]]
; CHECK-LLVM: %[[#R:]] = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 syncscope("singlethread") seq_cst
; CHECK-LLVM-NEXT: ret i32 %[[#R]]
define spir_func i32 @test_udec_wrap_singlethread_scope() {
entry:
  %r = atomicrmw udec_wrap ptr addrspace(1) @ui, i32 42 syncscope("singlethread") seq_cst
  ret i32 %r
}

; CHECK-SPIRV: FunctionCall [[#Int]] [[#]] [[#UIncFn]] [[#]] [[#Scope_CrossDevice]] [[#MemSem_Relaxed]] [[#Value]] [[#False]] [[#False]]
; CHECK-SPIRV: FunctionCall [[#Int]] [[#]] [[#UIncFnLocal]] [[#]] [[#Scope_CrossDevice]] [[#MemSem_Relaxed_Local]] [[#Value]] [[#False]] [[#False]]
; CHECK-LLVM: %[[#G:]] = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 monotonic
; CHECK-LLVM: %[[#L:]] = atomicrmw uinc_wrap ptr addrspace(3) @lui, i32 42 monotonic
; CHECK-LLVM: add i32 %[[#G]], %[[#L]]
define spir_func i32 @test_uinc_wrap_mixed_addrspace() {
entry:
  %g = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 monotonic
  %l = atomicrmw uinc_wrap ptr addrspace(3) @lui, i32 42 monotonic
  %s = add i32 %g, %l
  ret i32 %s
}

; CHECK-SPIRV: FunctionCall [[#Int]] [[#]] [[#UIncFn]] [[#]] [[#Scope_CrossDevice]] [[#MemSem_Relaxed]] [[#Value]] [[#False]] [[#False]]
; CHECK-SPIRV: FunctionCall [[#Long]] [[#]] [[#UIncFn64]] [[#]] [[#Scope_CrossDevice]] [[#MemSem_Relaxed]] [[#Value64]] [[#False]] [[#False]]
; CHECK-LLVM: %[[#A:]] = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 monotonic
; CHECK-LLVM: %[[#B:]] = atomicrmw uinc_wrap ptr addrspace(1) @ul, i64 42 monotonic
; CHECK-LLVM: %az = zext i32 %[[#A]] to i64
; CHECK-LLVM: add i64 %az, %[[#B]]
define spir_func i64 @test_uinc_wrap_mixed_width() {
entry:
  %a = atomicrmw uinc_wrap ptr addrspace(1) @ui, i32 42 monotonic
  %b = atomicrmw uinc_wrap ptr addrspace(1) @ul, i64 42 monotonic
  %az = zext i32 %a to i64
  %s = add i64 %az, %b
  ret i64 %s
}
