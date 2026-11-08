; Check that the proper error is reported for atomics wider than 64 bits.
;
; The writer stops at the first unsupported instruction, so each operation is
; translated from its own copy of the module.

; RUN: sed 's/OP/uinc_wrap/' %s > %t.ll
; RUN: not llvm-spirv %t.ll -o %t.spv 2>&1 | FileCheck %s --check-prefix=CHECK-UINC
; RUN: sed 's/OP/udec_wrap/' %s > %t.ll
; RUN: not llvm-spirv %t.ll -o %t.spv 2>&1 | FileCheck %s --check-prefix=CHECK-UDEC

; CHECK-UINC: Atomic uinc_wrap is not supported in SPIR-V!
; CHECK-UDEC: Atomic udec_wrap is not supported in SPIR-V!

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv64-amd-amdhsa"

@uv4i32 = common dso_local addrspace(1) global <4 x i32> zeroinitializer, align 16

define dso_local spir_func void @oversized_vector() local_unnamed_addr {
entry:
  %r = atomicrmw OP ptr addrspace(1) @uv4i32, <4 x i32> splat (i32 42) seq_cst
  ret void
}
