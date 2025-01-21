; This test checks that we don't face an assertion during the reverse translation:
;   Assertion `C->getType() == Ty->getElementType() && "Wrong type in array element initializer"' failed
; It also verifies that all the different address spaces were casted to the "common" one.

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_fpga_memory_attributes
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc
; RUN: FileCheck < %t.rev.ll %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%class.test = type { %"class.test_private" }
%"class.test_private" = type { i16 }

; CHECK: @llvm.global.annotations = appending global [2 x { ptr, ptr, ptr, i32, ptr }] [{ ptr, ptr, ptr, i32, ptr } { ptr addrspacecast (ptr addrspace(1) @samples to ptr), ptr @0, ptr undef, i32 undef, ptr undef }, { ptr, ptr, ptr, i32, ptr } { ptr addrspacecast (ptr addrspace(2) @foo to ptr), ptr @1, ptr undef, i32 undef, ptr undef }]

@llvm.global.annotations = addrspace(1) global [2 x { ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1) }] [ { ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1) } { ptr addrspace(1) @samples, ptr addrspace(1) @.str, ptr addrspace(1) null, i32 92, ptr addrspace(1) null }, { ptr addrspace(1), ptr addrspace(1), ptr addrspace(1), i32, ptr addrspace(1) } { ptr addrspace(1) addrspacecast (ptr addrspace(2) @foo to ptr addrspace(1)), ptr addrspace(1) @.str.2, ptr addrspace(1) null, i32 30, ptr addrspace(1) null }]
@samples = external addrspace(1) global { [64 x %class.test] }
@foo = dso_local addrspace(2) constant i32 1, align 4
@.str = addrspace(1) constant [13 x i8] c"{register:1}\00"
@.str.2 = addrspace(1) constant [13 x i8] c"{register:0}\00"
