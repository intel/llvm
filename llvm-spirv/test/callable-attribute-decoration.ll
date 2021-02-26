; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_INTEL_fast_composite
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -o %t.bc -r
; RUN: llvm-dis %t.bc -o %t.ll
; RUN: FileCheck < %t.ll %s --check-prefix=CHECK-LLVM
target triple = "spir64"


define dso_local <4 x i32> @foo(<4 x i32> %a, <4 x i32> %b) #0 {
entry:
  ret <4 x i32> %a
}
; CHECK-SPIRV: 3 Decorate {{[0-9]+}} CallableFunctionINTEL
; CHECK-LLVM: attributes
; CHECK-LLVM-SAME: "VCCallable"

attributes #0 = { "VCCallable" "VCFunction" }
