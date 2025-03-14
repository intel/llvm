; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fpga_memory_accesses -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: Capability FPGAMemoryAccessesINTEL
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_fpga_memory_accesses"
; CHECK-SPIRV: Decorate [[#DecTarget:]] BurstCoalesceINTEL
; CHECK-SPIRV: FunctionCall [[#]] [[#DecTarget]]

; CHECK-LLVM: [[#GV:]] = private unnamed_addr constant [11 x i8] c"{params:1}\00"
; CHECK-LLVM: %[[Call:[a-z0-9_.]+]] = call spir_func ptr addrspace(4) @accessor
; CHECK-LLVM: %[[#Ann:]] = call ptr addrspace(4) @llvm.ptr.annotation.p4.p0(ptr addrspace(4) %call, ptr @[[#GV]], ptr poison, i32 poison, ptr poison)
; CHECK-LLVM: call spir_func ptr addrspace(4) @_ZN8MyStructaSERKS_(ptr addrspace(4) %[[#Ann]]

; ModuleID = 'test.bc'
source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%struct.MyStruct = type { i32 }

$_ZN8MyStructaSERKS_ = comdat any

$accessor = comdat any

@.str.1 = private unnamed_addr addrspace(1) constant [14 x i8] c"<invalid loc>\00", section "llvm.metadata"
@.str.2 = private unnamed_addr addrspace(1) constant [11 x i8] c"{params:1}\00", section "llvm.metadata"

define spir_func void @foo(ptr %Ptr, ptr byval(%struct.MyStruct) align 4 %Val) {
entry:
  %Ptr.ascast = addrspacecast ptr %Ptr to ptr addrspace(4)
  %Val.ascast = addrspacecast ptr %Val to ptr addrspace(4)
  %call = call spir_func noundef ptr addrspace(4) @accessor(ptr addrspace(4) %Ptr.ascast)
  %0 = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) %call, ptr addrspace(1) @.str.2, ptr addrspace(1) @.str.1, i32 0, ptr addrspace(1) null)
  %call1 = call spir_func ptr addrspace(4) @_ZN8MyStructaSERKS_(ptr addrspace(4) %0, ptr addrspace(4) %Val.ascast)
  ret void
}

declare ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4), ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1))

declare spir_func ptr addrspace(4) @_ZN8MyStructaSERKS_(ptr addrspace(4) %this, ptr addrspace(4) %op)

declare spir_func ptr addrspace(4) @accessor(ptr addrspace(4) %this)
