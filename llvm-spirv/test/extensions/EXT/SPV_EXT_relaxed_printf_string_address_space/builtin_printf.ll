; Test that calls to "printf" are mapped to OpenCL Extended instruction "printf"
; Also ensure that spirv-val can validate format strings in non-constant space
;
; Testcase derived from:
;   #include <sycl/sycl.hpp>
;   int main() {
;     sycl::queue queue;
;     queue.submit([&](sycl::handler &cgh) {
;       cgh.single_task([] {
;         __builtin_printf("%s, %s %d %d %d %s!\n", "Hello", "world", 1, 2, 3, "Bam");
;       });
;     });
;   }

; RUN: llvm-as %s -o %t.bc
; RUN: not llvm-spirv %t.bc -o %t.spv 2>&1 | FileCheck %s --check-prefix=CHECK-WO-EXT

; RUN: llvm-spirv -spirv-text %t.bc -o %t.spt --spirv-ext=+SPV_EXT_relaxed_printf_string_address_space
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_EXT_relaxed_printf_string_address_space
; Change TODO to RUN when spirv-val allows non-constant printf formats
; TODO: spirv-val %t.spv


; CHECK-WO-EXT: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-WO-EXT: SPV_EXT_relaxed_printf_string_address_space extension should be allowed to translate this module, because this LLVM module contains the printf function with format string, whose address space is not equal to 2 (constant).

; CHECK-SPIRV: Extension "SPV_EXT_relaxed_printf_string_address_space"
; CHECK-SPIRV: ExtInst [[#]] [[#]] [[#]] printf [[#]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

@.str = external addrspace(1) constant [21 x i8]

define spir_kernel void @_ZTSZZ4mainENKUlRN4sycl3_V17handlerEE_clES2_EUlvE_() {
entry:
  %call.i = tail call spir_func i32 (ptr addrspace(4), ...) @printf(ptr addrspace(4) addrspacecast (ptr addrspace(1) @.str to ptr addrspace(4)), ptr addrspace(4) null, ptr addrspace(4) null, i32 0, i32 0, i32 0, ptr addrspace(4) null)
  ret void
}

declare spir_func i32 @printf(ptr addrspace(4), ...)
