;
; This test case checks that LLVM -> SPIR-V -> LLVM translation
; produces valid LLVM IR, where intrinsic global variables
; llvm.global_ctors and llvm.global_dtors, defined with non-default
; address space have correct (appending) linkage.
;
; No additional checks are needed in addition to simple translation
; to and from SPIR-V. In case of an error newly produced LLVM module
; validation would fail with the message:
;
; "Fails to verify module: invalid linkage for intrinsic global variable".
;
;
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_function_pointers -o %t.spv
; RUN: llvm-spirv -r %t.spv

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@llvm.global_ctors = appending addrspace(1) global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_test.cpp.ctor, ptr null }]
@llvm.global_dtors = appending addrspace(1) global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_test.cpp.dtor, ptr null }]

; Function Attrs: nounwind sspstrong
define internal void @_GLOBAL__sub_I_test.cpp.ctor() #0 {
  ret void
}

; Function Attrs: nounwind sspstrong
define internal void @_GLOBAL__sub_I_test.cpp.dtor() #0 {
  ret void
}

attributes #0 = { nounwind sspstrong }
